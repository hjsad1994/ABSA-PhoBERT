"""
PhoBERT Multi-Label ABSA Training Script

Features:
- Multi-label classification (33 binary labels: 11 aspects × 3 sentiments)
- FP16 mixed precision training
- 8-bit AdamW optimizer (memory efficient)
- Cosine learning rate scheduler
- Early stopping
- Multiple loss functions (BCE, Focal Loss, Weighted BCE)
- Epoch-based progress display with tqdm
- Training logs to file
- Reproducible results with fixed seeds
"""

import os
import sys
import shutil
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from datetime import datetime
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    set_seed
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    hamming_loss
)
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import custom loss functions
from focal_loss import (
    FocalLoss,
    WeightedBCEWithLogitsLoss,
    calculate_pos_weights,
    calculate_focal_alpha
)

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
        self.terminal_encoding = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        # Handle encoding errors for Windows console
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback: encode with error handling for Windows console
            safe_message = message.encode(self.terminal_encoding, errors='replace').decode(self.terminal_encoding)
            self.terminal.write(safe_message)
        
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

class MultiLabelABSADataset(Dataset):
    """
    Dataset for multi-label ABSA task
    
    Each sample consists of:
    - sentence: Vietnamese text
    - labels: 33 binary labels (0 or 1)
    """
    
    def __init__(self, sentences, labels, tokenizer, max_length):
        """
        Args:
            sentences (list): List of sentences
            labels (ndarray): Binary labels, shape (N, 33)
            tokenizer: PhoBERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute multi-label metrics
    
    Args:
        predictions (ndarray): Logits from model, shape (N, C)
        labels (ndarray): Binary labels, shape (N, C)
        threshold (float): Classification threshold
    
    Returns:
        dict: Dictionary of metrics
    """
    # Apply sigmoid to convert logits to probabilities
    probs = 1 / (1 + np.exp(-predictions))
    
    # Apply threshold to get binary predictions
    preds = (probs >= threshold).astype(int)
    
    # Micro average (treats all labels equally)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0
    )
    
    # Macro average (average across labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    # Hamming loss (fraction of incorrect labels)
    hamming = hamming_loss(labels, preds)
    
    # Exact match ratio (all labels must be correct)
    exact_match = np.mean(np.all(labels == preds, axis=1))
    
    return {
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'exact_match': exact_match
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_data(file_path, num_labels=33):
    """
    Load multi-label data from CSV
    
    Args:
        file_path (str): Path to CSV file
        num_labels (int): Number of labels (default: 33)
    
    Returns:
        tuple: (sentences, labels)
            - sentences: list of strings
            - labels: numpy array of shape (N, 33)
    """
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Extract sentences
    sentences = df['sentence'].tolist()
    
    # Extract labels
    label_cols = [f'label_{i}' for i in range(num_labels)]
    labels = df[label_cols].values.astype(float)
    
    # Calculate statistics
    total_active = labels.sum()
    avg_labels = total_active / len(labels)
    
    logger.info(f"  Loaded {len(sentences)} samples")
    logger.info(f"  Total active labels: {int(total_active)}")
    logger.info(f"  Average labels per sample: {avg_labels:.2f}")
    
    return sentences, labels


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
        str: Path to training file
    """
    use_oversampled = config['data'].get('use_oversampled_file', False)
    
    if use_oversampled:
        train_file = os.path.join(script_dir, 'data', 'train_oversampled.csv')
        
        if os.path.exists(train_file):
            logger.info(f"Using pre-oversampled file: {train_file}")
            return train_file
        else:
            logger.warning(f"Oversampled file not found: {train_file}")
            logger.warning("Falling back to original training file")
    
    # Use original training file
    train_file = os.path.join(script_dir, config['paths']['train_file'])
    logger.info(f"Using original training file: {train_file}")
    return train_file


def create_loss_function(config, train_labels, device):
    """
    Create loss function based on configuration
    
    Args:
        config (dict): Configuration dictionary
        train_labels (ndarray): Training labels for weight calculation
        device: Device to place loss function weights on
    
    Returns:
        tuple: (loss_fn, loss_name_str)
    """
    loss_type = config['training'].get('loss_type', 'bce')
    
    logger.info(f"Loss function: {loss_type}")
    
    if loss_type == 'focal':
        # Focal Loss
        focal_alpha_config = config['training'].get('focal_alpha', 0.25)
        focal_gamma = config['training'].get('focal_gamma', 2.0)
        
        # Auto-calculate alpha if specified
        if focal_alpha_config == "auto" or focal_alpha_config == "AUTO":
            logger.info("  Calculating optimal focal_alpha from training data...")
            focal_alpha = calculate_focal_alpha(train_labels, task_type="multi_label")
            logger.info(f"  Calculated alpha: {focal_alpha:.4f}")
        else:
            focal_alpha = float(focal_alpha_config)
        
        logger.info(f"  Focal Loss: alpha={focal_alpha:.4f}, gamma={focal_gamma}")
        
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss_name = f"Focal Loss (alpha={focal_alpha:.4f}, gamma={focal_gamma})"
        
    elif loss_type == 'weighted_bce':
        # Weighted BCE
        logger.info("  Calculating pos_weights from training data...")
        pos_weights = calculate_pos_weights(train_labels).to(device)
        logger.info(f"  Pos weights range: {pos_weights.min():.2f} - {pos_weights.max():.2f}")
        
        loss_fn = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
        loss_name = "Weighted BCE Loss"
        
    else:
        # Standard BCE
        logger.info("  BCEWithLogitsLoss (default)")
        loss_fn = nn.BCEWithLogitsLoss()
        loss_name = "BCE Loss"
    
    return loss_fn, loss_name


def create_optimizer(model, config):
    """
    Create optimizer
    
    Args:
        model: Model to optimize
        config (dict): Configuration dictionary
    
    Returns:
        Optimizer instance
    """
    optim_name = config['training']['optim']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optim_name == "adamw_bnb_8bit":
        # 8-bit AdamW (memory efficient)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
                eps=config['training']['adam_epsilon'],
                weight_decay=weight_decay
            )
            logger.info(f"✓ Optimizer: 8-bit AdamW (memory efficient)")
        except ImportError:
            logger.warning("  bitsandbytes not found, using standard AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
                eps=config['training']['adam_epsilon'],
                weight_decay=weight_decay
            )
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
            eps=config['training']['adam_epsilon'],
            weight_decay=weight_decay
        )
        logger.info(f"✓ Optimizer: AdamW")
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config (dict): Configuration dictionary
        num_training_steps (int): Total number of training steps
    
    Returns:
        Scheduler instance
    """
    warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"✓ Scheduler: Cosine with warmup ({warmup_steps} steps)")
    
    return scheduler


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, scaler, device, epoch, use_fp16=True):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for FP16
        device: Device to train on
        epoch (int): Current epoch number
        use_fp16 (bool): Whether to use FP16
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with/without FP16
        if use_fp16:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update scheduler
        scheduler.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, loss_fn, device, use_fp16=True, desc="Validation"):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        loss_fn: Loss function
        device: Device to evaluate on
        use_fp16 (bool): Whether to use FP16
        desc (str): Description for progress bar
    
    Returns:
        tuple: (avg_loss, all_predictions, all_labels)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=desc)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if use_fp16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            
            # Accumulate results
            total_loss += loss.item()
            all_predictions.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_predictions, all_labels


def save_results(model, tokenizer, test_predictions, test_labels, test_sentences, 
                config, paths, best_epoch, best_val_f1, training_time):
    """
    Save training results, evaluation report, and predictions
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_predictions (ndarray): Test predictions (logits)
        test_labels (ndarray): Test labels
        test_sentences (list): Test sentences
        config (dict): Configuration dictionary
        paths (dict): Dictionary of output paths
        best_epoch (int): Best epoch number
        best_val_f1 (float): Best validation F1 score
        training_time (float): Total training time in seconds
    """
    # Compute test metrics
    test_metrics = compute_metrics(test_predictions, test_labels, threshold=0.5)
    
    # Apply sigmoid and threshold
    probs = 1 / (1 + np.exp(-test_predictions))
    preds = (probs >= 0.5).astype(int)
    
    # Save evaluation report
    logger.info("=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)
    
    with open(paths['eval_report'], 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PhoBERT Multi-Label ABSA Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Test samples: {len(test_sentences)}\n")
        f.write(f"Num labels: {config['model']['num_labels']}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val F1 (micro): {best_val_f1:.4f}\n")
        f.write(f"Training time: {training_time:.2f}s\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  F1 (micro): {test_metrics['f1_micro']:.4f}\n")
        f.write(f"  F1 (macro): {test_metrics['f1_macro']:.4f}\n")
        f.write(f"  Precision (micro): {test_metrics['precision_micro']:.4f}\n")
        f.write(f"  Recall (micro): {test_metrics['recall_micro']:.4f}\n")
        f.write(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}\n")
        f.write(f"  Exact Match: {test_metrics['exact_match']:.4f}\n")
    
    logger.info(f"Evaluation report saved: {paths['eval_report']}")
    
    # Save predictions
    pred_df = pd.DataFrame({'sentence': test_sentences})
    
    # Add predicted labels
    for i in range(config['model']['num_labels']):
        pred_df[f'pred_label_{i}'] = preds[:, i]
    
    pred_df.to_csv(paths['predictions'], index=False, encoding='utf-8')
    logger.info(f"Predictions saved: {paths['predictions']}")
    
    return test_metrics


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
    print("PhoBERT Multi-Label ABSA Training")
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
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Loading Data")
    logger.info("=" * 80)
    
    # Get training file
    train_file = get_training_file(script_dir, config)
    
    # Load datasets
    train_sentences, train_labels = load_data(train_file, config['model']['num_labels'])
    val_sentences, val_labels = load_data(
        os.path.join(script_dir, config['paths']['validation_file']),
        config['model']['num_labels']
    )
    test_sentences, test_labels = load_data(
        os.path.join(script_dir, config['paths']['test_file']),
        config['model']['num_labels']
    )
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = MultiLabelABSADataset(
        train_sentences, train_labels, tokenizer, config['model']['max_length']
    )
    val_dataset = MultiLabelABSADataset(
        val_sentences, val_labels, tokenizer, config['model']['max_length']
    )
    test_dataset = MultiLabelABSADataset(
        test_sentences, test_labels, tokenizer, config['model']['max_length']
    )
    
    logger.info(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Create dataloaders
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    num_workers = config['training']['dataloader_num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['training']['dataloader_pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['training']['dataloader_pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['training']['dataloader_pin_memory']
    )
    
    logger.info(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print()
    
    # =========================================================================
    # 3. TRAINING CONFIGURATION
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function
    loss_fn, loss_name = create_loss_function(config, train_labels, device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Calculate training steps
    num_epochs = config['training']['num_train_epochs']
    num_training_steps = len(train_loader) * num_epochs
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Create gradient scaler for FP16
    use_fp16 = config['training']['fp16']
    scaler = GradScaler() if use_fp16 else None
    
    # Training settings
    effective_batch = batch_size * config['training']['gradient_accumulation_steps']
    logger.info(f"✓ Effective batch size: {effective_batch}")
    logger.info(f"✓ Learning rate: {config['training']['learning_rate']}")
    logger.info(f"✓ Epochs: {num_epochs}")
    logger.info(f"✓ FP16: {use_fp16}")
    logger.info(f"✓ Loss: {loss_name}")
    print()
    
    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    logger.info("=" * 80)
    logger.info("🎯 STARTING TRAINING")
    logger.info("=" * 80)
    print()
    
    # Early stopping parameters
    best_val_f1 = 0.0
    best_epoch = 0
    best_checkpoint_name = None
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    # Training start time
    import time
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, 
            scaler, device, epoch, use_fp16
        )
        
        # Evaluate on validation set
        val_loss, val_predictions, val_labels_array = evaluate(
            model, val_loader, loss_fn, device, use_fp16, desc=f"Validation {epoch}"
        )
        
        # Compute validation metrics
        val_metrics = compute_metrics(val_predictions, val_labels_array, threshold=0.5)
        
        # Log epoch results
        print()
        logger.info(f"Epoch {epoch}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val F1 (micro): {val_metrics['f1_micro']:.4f}")
        logger.info(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Check if best model
        if val_metrics['f1_micro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_micro']
            best_epoch = epoch
            patience_counter = 0
            
            # Calculate F1-based checkpoint name
            f1_int = int(best_val_f1 * 10000)
            checkpoint_name = f'checkpoint-{f1_int}'
            checkpoint_path = os.path.join(paths['output_dir'], checkpoint_name)
            
            # Remove old best checkpoint if exists
            if best_checkpoint_name is not None:
                old_checkpoint_path = os.path.join(paths['output_dir'], best_checkpoint_name)
                if os.path.exists(old_checkpoint_path):
                    try:
                        shutil.rmtree(old_checkpoint_path)
                        logger.info(f"  🗑️  Removed old checkpoint: {best_checkpoint_name}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Could not remove old checkpoint: {e}")
            
            # Save new best checkpoint
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            best_checkpoint_name = checkpoint_name
            
            logger.info(f"  ✓ New best model saved! (F1: {best_val_f1:.4f})")
            logger.info(f"  📁 Checkpoint: {checkpoint_name}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement (patience: {patience_counter}/{patience})")
        
        print()
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best epoch: {best_epoch} with F1: {best_val_f1:.4f}")
            logger.info(f"Best checkpoint: {best_checkpoint_name}")
            break
    
    # Training time
    training_time = time.time() - training_start_time
    
    # Log training results
    print()
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✓ Total training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
    logger.info(f"✓ Best epoch: {best_epoch}")
    logger.info(f"✓ Best val F1 (micro): {best_val_f1:.4f}")
    print()
    
    # Load best model for evaluation
    logger.info("Loading best model for test evaluation...")
    best_checkpoint = os.path.join(paths['output_dir'], best_checkpoint_name)
    logger.info(f"Loading from: {best_checkpoint_name}")
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint).to(device)
    print()
    
    # =========================================================================
    # 5. EVALUATION ON TEST SET
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 80)
    
    test_loss, test_predictions, test_labels_array = evaluate(
        model, test_loader, loss_fn, device, use_fp16, desc="Testing"
    )
    
    # Compute test metrics
    test_metrics = compute_metrics(test_predictions, test_labels_array, threshold=0.5)
    
    print()
    logger.info(f"✓ Test Loss: {test_loss:.4f}")
    logger.info(f"✓ Test F1 (micro): {test_metrics['f1_micro']:.4f}")
    logger.info(f"✓ Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"✓ Test Precision (micro): {test_metrics['precision_micro']:.4f}")
    logger.info(f"✓ Test Recall (micro): {test_metrics['recall_micro']:.4f}")
    logger.info(f"✓ Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    logger.info(f"✓ Test Exact Match: {test_metrics['exact_match']:.4f}")
    print()
    
    # =========================================================================
    # 6. SAVE RESULTS
    # =========================================================================
    
    # Copy best checkpoint to 'best_model' folder for convenience
    best_model_path = os.path.join(paths['output_dir'], 'best_model')
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(best_checkpoint, best_model_path)
    logger.info(f"Best model also saved to: best_model/")
    print()
    
    test_metrics = save_results(
        model, tokenizer, test_predictions, test_labels_array, test_sentences,
        config, paths, best_epoch, best_val_f1, training_time
    )
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print()
    logger.info("=" * 80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 80)
    print()
    logger.info("✓ Summary:")
    logger.info(f"   • Model fine-tuned successfully")
    logger.info(f"   • Best epoch: {best_epoch}/{num_epochs}")
    logger.info(f"   • Best val F1: {best_val_f1:.4f}")
    logger.info(f"   • Training time: {training_time/60:.2f} minutes")
    logger.info(f"   • Test F1 (micro): {test_metrics['f1_micro']:.4f}")
    logger.info(f"   • Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"   • Best checkpoint: {best_checkpoint_name}")
    logger.info(f"   • Best model also in: best_model/")
    logger.info(f"   • Evaluation report: {paths['eval_report']}")
    logger.info(f"   • Predictions: {paths['predictions']}")
    print()
    logger.info(f"📝 Training log saved: {log_file_path}")
    print()
    
    # =========================================================================
    # 8. CLEANUP
    # =========================================================================
    sys.stdout = tee_logger.terminal
    sys.stderr = tee_logger.terminal
    tee_logger.close()


if __name__ == '__main__':
    main()
