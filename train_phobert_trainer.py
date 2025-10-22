"""
PhoBERT Training with HuggingFace Trainer API
Supports: FP16, 8-bit optimizer, cosine scheduler, per-aspect oversampling
"""
import os
import yaml
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Class
# ============================================================================

class ABSADataset(Dataset):
    """Dataset for ABSA task with sentence-aspect pairs"""
    
    def __init__(self, sentences, aspects, sentiments, tokenizer, max_length):
        self.sentences = sentences
        self.aspects = aspects
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        aspect = str(self.aspects[idx])
        sentiment = self.sentiments[idx]
        
        # Combine sentence and aspect with special separator
        # PhoBERT uses </s></s> as separator
        text = f"{sentence} </s></s> {aspect}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# Data Loading with Oversampling
# ============================================================================

def oversample_per_aspect(df, aspect_col='aspect', label_col='sentiment'):
    """
    Oversample data per aspect: balance labels within each aspect
    """
    logger.info("Applying per-aspect oversampling...")
    
    unique_aspects = df[aspect_col].unique()
    oversampled_dfs = []
    
    for aspect in unique_aspects:
        aspect_df = df[df[aspect_col] == aspect].copy()
        
        # Count labels for this aspect
        label_counts = aspect_df[label_col].value_counts()
        max_count = label_counts.max()
        
        logger.info(f"  Aspect '{aspect}': {dict(label_counts)} → balancing to {max_count}")
        
        aspect_samples = []
        
        for label in label_counts.index:
            label_samples = aspect_df[aspect_df[label_col] == label]
            count = len(label_samples)
            
            if count < max_count:
                # Oversample
                n_samples_needed = max_count - count
                oversampled = label_samples.sample(n=n_samples_needed, replace=True, random_state=42)
                aspect_samples.append(pd.concat([label_samples, oversampled], ignore_index=True))
            else:
                aspect_samples.append(label_samples)
        
        aspect_balanced = pd.concat(aspect_samples, ignore_index=True)
        oversampled_dfs.append(aspect_balanced)
    
    result_df = pd.concat(oversampled_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Total: {len(df)} → {len(result_df)} samples after oversampling")
    
    return result_df


def load_data(file_path, sentiment_mapping, text_col, aspect_col, label_col, apply_oversampling=False):
    """Load data from CSV and convert sentiments to numeric labels"""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check required columns
    required_cols = [text_col, aspect_col, label_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")
    
    # Apply oversampling if requested (only for training data)
    if apply_oversampling:
        df = oversample_per_aspect(df, aspect_col, label_col)
    
    # Convert sentiment to numeric
    df['label'] = df[label_col].map(sentiment_mapping)
    
    # Check for unmapped sentiments
    if df['label'].isna().any():
        unmapped = df[df['label'].isna()][label_col].unique()
        raise ValueError(f"Found unmapped sentiments: {unmapped}")
    
    logger.info(f"  Loaded {len(df)} samples")
    logger.info(f"  Distribution: {df[label_col].value_counts().to_dict()}")
    
    return df[text_col].tolist(), df[aspect_col].tolist(), df['label'].astype(int).tolist()


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Load config
    print("=" * 80)
    print("PhoBERT ABSA Training with HuggingFace Trainer")
    print("=" * 80)
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config['general']['seed']
    set_seed(seed)
    logger.info(f"Seed: {seed}")
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['evaluation_report']), exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Sentiment mapping
    sentiment_mapping = config['sentiment_labels']
    label_to_sentiment = {v: k for k, v in sentiment_mapping.items()}
    
    logger.info(f"Sentiment mapping: {sentiment_mapping}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        problem_type="single_label_classification",
        use_safetensors=True
    )
    
    logger.info(f"Model loaded with {config['model']['num_labels']} labels")
    
    # Load data
    logger.info("=" * 80)
    logger.info("Loading Data")
    logger.info("=" * 80)
    
    train_sentences, train_aspects, train_labels = load_data(
        config['paths']['train_file'],
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=config['data']['oversampling']
    )
    
    val_sentences, val_aspects, val_labels = load_data(
        config['paths']['validation_file'],
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=False
    )
    
    test_sentences, test_aspects, test_labels = load_data(
        config['paths']['test_file'],
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=False
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ABSADataset(
        train_sentences, train_aspects, train_labels,
        tokenizer, config['model']['max_length']
    )
    val_dataset = ABSADataset(
        val_sentences, val_aspects, val_labels,
        tokenizer, config['model']['max_length']
    )
    test_dataset = ABSADataset(
        test_sentences, test_aspects, test_labels,
        tokenizer, config['model']['max_length']
    )
    
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Training arguments
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    
    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        
        # Batch size and accumulation
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        
        # Optimizer
        optim=config['training']['optim'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],
        
        # Scheduler
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        
        # Training duration
        num_train_epochs=config['training']['num_train_epochs'],
        
        # Mixed precision
        fp16=config['training']['fp16'],
        fp16_opt_level=config['training']['fp16_opt_level'],
        fp16_full_eval=config['training']['fp16_full_eval'],
        tf32=config['training']['tf32'],
        
        # DataLoader
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        dataloader_prefetch_factor=config['training']['dataloader_prefetch_factor'],
        dataloader_persistent_workers=config['training']['dataloader_persistent_workers'],
        
        # Memory optimization
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        auto_find_batch_size=config['training']['auto_find_batch_size'],
        group_by_length=config['training']['group_by_length'],
        
        # Evaluation & checkpointing
        eval_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        
        # Logging
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        logging_dir=config['paths']['log_dir'],
        report_to=config['training']['report_to'],
        
        # Reproducibility
        seed=config['training']['seed'],
        data_seed=config['training']['data_seed'],
        
        # Misc
        disable_tqdm=config['training']['disable_tqdm'],
        prediction_loss_only=config['training']['prediction_loss_only'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        label_names=config['training']['label_names'],
        include_inputs_for_metrics=config['training']['include_inputs_for_metrics'],
    )
    
    effective_batch = (
        config['training']['per_device_train_batch_size'] * 
        config['training']['gradient_accumulation_steps'] *
        (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"FP16: {config['training']['fp16']}")
    logger.info(f"Optimizer: {config['training']['optim']}")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config['training']['early_stopping_patience'],
                early_stopping_threshold=config['training']['early_stopping_threshold']
            )
        ]
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("Training")
    logger.info("=" * 80)
    
    trainer.train()
    
    # Evaluate on test set
    logger.info("=" * 80)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 80)
    
    test_results = trainer.predict(test_dataset)
    test_metrics = test_results.metrics
    
    logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['test_precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['test_recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['test_f1']:.4f}")
    
    # Detailed classification report
    predictions = np.argmax(test_results.predictions, axis=1)
    labels = test_results.label_ids
    
    report = classification_report(
        labels,
        predictions,
        target_names=[label_to_sentiment[i] for i in range(config['model']['num_labels'])],
        digits=4
    )
    
    logger.info("\nClassification Report:")
    print(report)
    
    # Save results
    logger.info("=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)
    
    # Save evaluation report
    report_path = config['paths']['evaluation_report']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PhoBERT ABSA Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  Accuracy: {test_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['test_precision']:.4f}\n")
        f.write(f"  Recall: {test_metrics['test_recall']:.4f}\n")
        f.write(f"  F1: {test_metrics['test_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    logger.info(f"Evaluation report saved: {report_path}")
    
    # Save predictions
    pred_path = config['paths']['predictions_file']
    pred_df = pd.DataFrame({
        'sentence': test_sentences,
        'aspect': test_aspects,
        'true_sentiment': [label_to_sentiment[l] for l in labels],
        'predicted_sentiment': [label_to_sentiment[p] for p in predictions],
        'correct': labels == predictions
    })
    pred_df.to_csv(pred_path, index=False, encoding='utf-8')
    logger.info(f"Predictions saved: {pred_path}")
    
    # Save best model
    best_model_path = os.path.join(config['paths']['output_dir'], 'best_model')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved: {best_model_path}")
    
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
