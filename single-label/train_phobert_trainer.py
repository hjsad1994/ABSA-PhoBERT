"""
PhoBERT Single-Label ABSA Training Script

Features:
- Single-label classification (3 classes: Negative, Neutral, Positive)
- Sentence-aspect pair format
- FP16 mixed precision training
- 8-bit AdamW optimizer (memory efficient)
- Cosine learning rate scheduler
- Per-aspect oversampling
- Checkpoint naming by F1 score
- Early stopping
- Training logs to file
- Reproducible results with fixed seeds
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from torch.utils.data import Dataset

# Import checkpoint renamer
from checkpoint_renamer import SimpleMetricCheckpointCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TeeLogger - Log to Both Console and File
# ============================================================================

class TeeLogger:
    """Logger that writes to both console and file simultaneously"""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logging(script_dir):
    """
    Setup logging to both console and file
    
    Args:
        script_dir: Directory where script is located
    
    Returns:
        tuple: (TeeLogger instance, log file path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(script_dir, "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"📝 Training log will be saved to: {log_file}\n")
    
    return tee, log_file


# ============================================================================
# Dataset Class
# ============================================================================

class ABSADataset(Dataset):
    """
    Dataset for ABSA task with sentence-aspect pairs
    
    Format:
    - Input: "[Sentence] </s></s> [Aspect]"
    - Output: Sentiment label (0=Negative, 1=Neutral, 2=Positive)
    """
    
    def __init__(self, sentences, aspects, sentiments, tokenizer, max_length):
        """
        Args:
            sentences (list): List of sentences
            aspects (list): List of aspects
            sentiments (list): List of sentiment labels (0, 1, or 2)
            tokenizer: PhoBERT tokenizer
            max_length (int): Maximum sequence length
        """
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
        
        # Combine sentence and aspect with PhoBERT separator
        # PhoBERT uses </s></s> as separator between segments
        text = f"{sentence} </s></s> {aspect}"
        
        # Tokenize
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
# Metrics Computation
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute classification metrics
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        dict: Dictionary of metrics (accuracy, precision, recall, f1)
    """
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
# Data Loading
# ============================================================================

def oversample_per_aspect(df, aspect_col='aspect', label_col='sentiment'):
    """
    Oversample data per aspect: balance sentiments within each aspect
    
    Example:
        Battery: Positive=100, Negative=50, Neutral=20
        → After: Positive=100, Negative=100, Neutral=100
    
    Args:
        df: DataFrame with columns: sentence, aspect, sentiment
        aspect_col: Name of aspect column
        label_col: Name of label column
    
    Returns:
        DataFrame: Oversampled and shuffled data
    """
    logger.info("Applying per-aspect oversampling...")
    
    unique_aspects = df[aspect_col].unique()
    oversampled_dfs = []
    
    for aspect in unique_aspects:
        aspect_df = df[df[aspect_col] == aspect].copy()
        
        # Count sentiments for this aspect
        label_counts = aspect_df[label_col].value_counts()
        max_count = label_counts.max()
        
        logger.info(f"  Aspect '{aspect}': {dict(label_counts)} → balancing to {max_count}")
        
        aspect_samples = []
        
        for label in label_counts.index:
            label_samples = aspect_df[aspect_df[label_col] == label]
            count = len(label_samples)
            
            if count < max_count:
                # Oversample: duplicate samples to reach max_count
                n_samples_needed = max_count - count
                oversampled = label_samples.sample(
                    n=n_samples_needed, 
                    replace=True, 
                    random_state=42
                )
                aspect_samples.append(pd.concat([label_samples, oversampled], ignore_index=True))
            else:
                aspect_samples.append(label_samples)
        
        aspect_balanced = pd.concat(aspect_samples, ignore_index=True)
        oversampled_dfs.append(aspect_balanced)
    
    # Combine all aspects and shuffle
    result_df = pd.concat(oversampled_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Total: {len(df)} → {len(result_df)} samples after oversampling")
    
    return result_df


def load_data(file_path, sentiment_mapping, text_col, aspect_col, label_col, apply_oversampling=False):
    """
    Load data from CSV and convert sentiments to numeric labels
    
    Args:
        file_path (str): Path to CSV file
        sentiment_mapping (dict): Mapping from sentiment name to label (e.g., {"Negative": 0})
        text_col (str): Name of text column
        aspect_col (str): Name of aspect column
        label_col (str): Name of label column
        apply_oversampling (bool): Whether to apply per-aspect oversampling
    
    Returns:
        tuple: (sentences, aspects, labels)
    """
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
    
    # Convert sentiment names to numeric labels
    df['label'] = df[label_col].map(sentiment_mapping)
    
    # Check for unmapped sentiments
    if df['label'].isna().any():
        unmapped = df[df['label'].isna()][label_col].unique()
        raise ValueError(f"Found unmapped sentiments: {unmapped}")
    
    logger.info(f"  Loaded {len(df)} samples")
    logger.info(f"  Distribution: {df[label_col].value_counts().to_dict()}")
    
    return (
        df[text_col].tolist(),
        df[aspect_col].tolist(),
        df['label'].astype(int).tolist()
    )


def load_config(script_dir):
    """
    Load configuration from config.yaml
    
    Args:
        script_dir (str): Directory where script is located
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_directories(script_dir, config):
    """
    Create necessary output directories
    
    Args:
        script_dir (str): Directory where script is located
        config (dict): Configuration dictionary
    
    Returns:
        dict: Dictionary of absolute paths
    """
    paths = {
        'output_dir': os.path.join(script_dir, config['paths']['output_dir']),
        'eval_report': os.path.join(script_dir, config['paths']['evaluation_report']),
        'predictions': os.path.join(script_dir, config['paths']['predictions_file']),
        'log_dir': os.path.join(script_dir, config['paths']['log_dir'])
    }
    
    # Create directories
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(paths['eval_report']), exist_ok=True)
    os.makedirs(os.path.dirname(paths['predictions']), exist_ok=True)
    os.makedirs(paths['log_dir'], exist_ok=True)
    
    return paths


def get_training_file(script_dir, config):
    """
    Determine which training file to use
    
    Args:
        script_dir (str): Directory where script is located
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_file_path, apply_oversampling_on_fly)
    """
    use_oversampled = config['data'].get('use_oversampled_file', False)
    
    if use_oversampled:
        train_file = os.path.join(script_dir, config['paths']['train_oversampled_file'])
        
        if os.path.exists(train_file):
            logger.info(f"Using pre-oversampled file: {train_file}")
            return train_file, False
        else:
            logger.warning(f"Oversampled file not found: {train_file}")
            logger.warning("Falling back to original file with on-the-fly oversampling")
    
    # Use original training file
    train_file = os.path.join(script_dir, config['paths']['train_file'])
    apply_oversampling = config['data'].get('oversampling', False)
    
    if apply_oversampling:
        logger.info(f"Using original file with on-the-fly oversampling: {train_file}")
    else:
        logger.info(f"Using original training file: {train_file}")
    
    return train_file, apply_oversampling


def save_results(trainer, test_dataset, test_sentences, test_aspects, config, 
                 paths, label_to_sentiment, train_result, test_metrics):
    """
    Save training results, evaluation report, and predictions
    
    Args:
        trainer: Trained model
        test_dataset: Test dataset
        test_sentences: Test sentences
        test_aspects: Test aspects
        config: Configuration dictionary
        paths: Dictionary of output paths
        label_to_sentiment: Mapping from label to sentiment name
        train_result: Training result
        test_metrics: Test metrics
    """
    # Get predictions
    test_results = trainer.predict(test_dataset)
    predictions = np.argmax(test_results.predictions, axis=1)
    labels = test_results.label_ids
    
    # Generate classification report
    report = classification_report(
        labels,
        predictions,
        target_names=[label_to_sentiment[i] for i in range(config['model']['num_labels'])],
        digits=4
    )
    
    logger.info("=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)
    
    # Save evaluation report
    with open(paths['eval_report'], 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PhoBERT Single-Label ABSA Evaluation Report\n")
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
    
    logger.info(f"Evaluation report saved: {paths['eval_report']}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'sentence': test_sentences,
        'aspect': test_aspects,
        'true_sentiment': [label_to_sentiment[l] for l in labels],
        'predicted_sentiment': [label_to_sentiment[p] for p in predictions],
        'correct': labels == predictions
    })
    pred_df.to_csv(paths['predictions'], index=False, encoding='utf-8')
    logger.info(f"Predictions saved: {paths['predictions']}")
    
    # Save best model
    best_model_path = os.path.join(paths['output_dir'], 'best_model')
    trainer.save_model(best_model_path)
    trainer.tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved: {best_model_path}")
    
    # Print classification report to console
    logger.info("\nClassification Report:")
    print(report)


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function"""
    
    # =========================================================================
    # 0. SETUP
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    tee_logger, log_file_path = setup_logging(script_dir)
    
    print("=" * 80)
    print("PhoBERT Single-Label ABSA Training")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config(script_dir)
    
    # Set seed for reproducibility
    seed = config.get('reproducibility', {}).get('training_seed', 42)
    set_seed(seed)
    logger.info(f"Random seed for training: {seed}")
    logger.info("(Ensures reproducible results for research)")
    print()
    
    # Create output directories
    paths = create_directories(script_dir, config)
    logger.info(f"Output directory: {paths['output_dir']}")
    logger.info(f"Log directory: {paths['log_dir']}")
    print()
    
    # Sentiment mapping
    sentiment_mapping = config['sentiment_labels']
    label_to_sentiment = {v: k for k, v in sentiment_mapping.items()}
    logger.info(f"Sentiment mapping: {sentiment_mapping}")
    print()
    
    # =========================================================================
    # 1. LOAD MODEL AND TOKENIZER
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Loading Model and Tokenizer")
    logger.info("=" * 80)
    
    logger.info(f"Model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        problem_type=config['model']['problem_type'],
        use_safetensors=True
    )
    
    logger.info(f"✓ Model loaded with {config['model']['num_labels']} labels")
    logger.info(f"✓ Problem type: {config['model']['problem_type']}")
    print()
    
    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Loading Data")
    logger.info("=" * 80)
    
    # Get training file
    train_file, apply_oversampling_on_fly = get_training_file(script_dir, config)
    
    # Load datasets
    train_sentences, train_aspects, train_labels = load_data(
        train_file,
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=apply_oversampling_on_fly
    )
    
    val_sentences, val_aspects, val_labels = load_data(
        os.path.join(script_dir, config['paths']['validation_file']),
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=False
    )
    
    test_sentences, test_aspects, test_labels = load_data(
        os.path.join(script_dir, config['paths']['test_file']),
        sentiment_mapping,
        config['data']['text_column'],
        config['data']['aspect_column'],
        config['data']['label_column'],
        apply_oversampling=False
    )
    
    # Create datasets
    logger.info("\nCreating datasets...")
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
    
    logger.info(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print()
    
    # =========================================================================
    # 3. TRAINING CONFIGURATION
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    
    training_args = TrainingArguments(
        output_dir=paths['output_dir'],
        
        # Batch size
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
        logging_dir=paths['log_dir'],
        report_to=config['training']['report_to'],
        
        # Reproducibility
        seed=config.get('reproducibility', {}).get('training_seed', seed),
        data_seed=config.get('reproducibility', {}).get('data_loader_seed', seed),
        
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
    
    logger.info(f"✓ Effective batch size: {effective_batch}")
    logger.info(f"✓ Learning rate: {config['training']['learning_rate']}")
    logger.info(f"✓ Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"✓ FP16: {config['training']['fp16']}")
    logger.info(f"✓ Optimizer: {config['training']['optim']}")
    print()
    
    # =========================================================================
    # 4. CREATE TRAINER WITH CALLBACKS
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Creating Trainer & Callbacks")
    logger.info("=" * 80)
    
    # Checkpoint renamer callback - rename by F1 score (4 digits)
    checkpoint_callback = SimpleMetricCheckpointCallback(
        metric_name='eval_f1', 
        multiply_by=10000
    )
    logger.info("✓ Checkpoint Renamer: Will rename checkpoints by F1 score")
    logger.info("  Example: F1=87.53% → checkpoint-8753")
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    logger.info(f"✓ Early Stopping: patience={config['training']['early_stopping_patience']}, "
                f"threshold={config['training']['early_stopping_threshold']}")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    logger.info("✓ Trainer created successfully")
    print()
    
    # =========================================================================
    # 5. TRAINING
    # =========================================================================
    logger.info("=" * 80)
    logger.info("🎯 STARTING TRAINING")
    logger.info("=" * 80)
    print()
    
    train_result = trainer.train()
    
    # Log training results
    print()
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✓ Training loss: {train_result.training_loss:.4f}")
    logger.info(f"✓ Training time: {train_result.metrics['train_runtime']:.2f}s")
    logger.info(f"✓ Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    logger.info(f"✓ Steps/second: {train_result.metrics['train_steps_per_second']:.2f}")
    print()
    
    # =========================================================================
    # 6. EVALUATION
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 80)
    
    test_results = trainer.predict(test_dataset)
    test_metrics = test_results.metrics
    
    logger.info(f"✓ Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"✓ Test Precision: {test_metrics['test_precision']:.4f}")
    logger.info(f"✓ Test Recall: {test_metrics['test_recall']:.4f}")
    logger.info(f"✓ Test F1: {test_metrics['test_f1']:.4f}")
    print()
    
    # =========================================================================
    # 7. SAVE RESULTS
    # =========================================================================
    save_results(
        trainer, test_dataset, test_sentences, test_aspects,
        config, paths, label_to_sentiment, train_result, test_metrics
    )
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print()
    logger.info("=" * 80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 80)
    print()
    logger.info("✓ Summary:")
    logger.info(f"   • Model fine-tuned successfully")
    logger.info(f"   • Training loss: {train_result.training_loss:.4f}")
    logger.info(f"   • Test F1: {test_metrics['test_f1']:.4f}")
    logger.info(f"   • Best model saved: {os.path.join(paths['output_dir'], 'best_model')}")
    logger.info(f"   • Evaluation report: {paths['eval_report']}")
    logger.info(f"   • Predictions: {paths['predictions']}")
    print()
    logger.info(f"📝 Training log saved: {log_file_path}")
    print()
    
    # =========================================================================
    # 9. CLEANUP
    # =========================================================================
    sys.stdout = tee_logger.terminal
    sys.stderr = tee_logger.terminal
    tee_logger.close()


if __name__ == '__main__':
    main()
