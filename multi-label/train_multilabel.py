"""
Training Script for Multi-Label ABSA
Train PhoBERT to predict all 11 aspects simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
from datetime import datetime

from model_multilabel import MultiLabelPhoBERT
from dataset_multilabel import MultiLabelABSADataset
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, optimizer, scheduler, device, focal_loss_fn):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, attention_mask)
        
        # Loss (Focal Loss)
        loss = focal_loss_fn(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device, aspect_names):
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Predict
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)  # [batch_size, num_aspects]
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics per aspect (ONLY for aspects with real sentiment, ignore class 3 = "none")
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i].numpy()
        aspect_labels = all_labels[:, i].numpy()
        
        # Create mask for valid labels (not "none")
        valid_mask = (aspect_labels != 3)
        
        if valid_mask.sum() == 0:
            # No real sentiment for this aspect in validation set
            aspect_metrics[aspect] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'count': 0
            }
            continue
        
        # Filter to only valid positions
        valid_preds = aspect_preds[valid_mask]
        valid_labels = aspect_labels[valid_mask]
        
        # Accuracy (on valid positions only)
        acc = accuracy_score(valid_labels, valid_preds)
        
        # Precision, Recall, F1 (only on 3 real classes: 0, 1, 2)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average='weighted', zero_division=0,
            labels=[0, 1, 2]  # Only evaluate on Negative, Neutral, Positive
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': valid_mask.sum()
        }
    
    # Overall metrics (average across aspects, weighted by valid count)
    valid_mask_all = (all_labels != 3)
    if valid_mask_all.sum() > 0:
        overall_acc = (all_preds[valid_mask_all] == all_labels[valid_mask_all]).float().mean().item()
    else:
        overall_acc = 0.0
    
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
    
    return {
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'per_aspect': aspect_metrics
    }

def print_metrics(metrics, epoch=None):
    """Pretty print metrics"""
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results")
        print(f"{'='*80}")
    
    print(f"\nOverall Metrics:")
    print(f"   Accuracy:  {metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {metrics['overall_f1']*100:.2f}%")
    print(f"   Precision: {metrics['overall_precision']*100:.2f}%")
    print(f"   Recall:    {metrics['overall_recall']*100:.2f}%")
    
    print(f"\nPer-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for aspect, m in metrics['per_aspect'].items():
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%")

def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"[OK] Saved best model: {best_path}")
    
    return checkpoint_path

def main(args):
    print("=" * 80)
    print("Multi-Label ABSA Training")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Get script directory for resolving relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert relative paths to absolute paths
    for key in ['train_file', 'validation_file', 'test_file']:
        if key in config['paths']:
            path = config['paths'][key]
            if not os.path.isabs(path):
                # Relative to script directory
                abs_path = os.path.join(script_dir, path)
                config['paths'][key] = abs_path
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[OK] Using device: {device}")
    
    # Set seed from reproducibility config
    seed = config['reproducibility']['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load tokenizer
    print(f"\n[OK] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load datasets
    print(f"\n[OK] Loading datasets...")
    train_dataset = MultiLabelABSADataset(
        config['paths']['train_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = MultiLabelABSADataset(
        config['paths']['validation_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_dataset = MultiLabelABSADataset(
        config['paths']['test_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 32)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 64)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # =====================================================================
    # SETUP FOCAL LOSS
    # =====================================================================
    print(f"\n{'='*80}")
    print("Setting up Focal Loss...")
    print(f"{'='*80}")
    
    # Read focal loss config
    focal_config = config.get('multi_label', {})
    use_focal_loss = focal_config.get('use_focal_loss', True)
    focal_gamma = focal_config.get('focal_gamma', 2.0)
    focal_alpha_config = focal_config.get('focal_alpha', 'auto')
    
    if not use_focal_loss:
        print(f"\n[WARNING]️  Focal Loss is DISABLED in config!")
        print(f"   Using standard CrossEntropyLoss")
        # Fallback to cross entropy (not recommended for imbalanced data)
        focal_loss_fn = None  # Will handle this in train_epoch
    else:
        sentiment_to_idx = config['sentiment_labels']
        
        # Determine alpha weights
        if focal_alpha_config == 'auto':
            print(f"\nAlpha mode: AUTO (global inverse frequency)")
            # Calculate alpha directly from loaded dataset
            print(f"\n[INFO] Calculating alpha from training dataset...")
            
            # Count sentiments in training data
            from collections import Counter
            all_sentiments = train_dataset.labels.flatten()
            unique, counts = np.unique(all_sentiments, return_counts=True)
            
            # Calculate total ONLY for real sentiments (exclude class 3 = "none")
            # This matches ViSoBERT behavior where we only have 3 classes
            total_all = counts.sum()
            mask_real = unique != 3
            total = counts[mask_real].sum()
            num_classes = 3  # negative, neutral, positive (exclude "none")
            
            # Calculate inverse frequency alpha for each sentiment class
            alpha = []
            sentiment_order = [0, 1, 2, 3]  # negative, neutral, positive, none
            
            print(f"\n   Total: {total:,} (real sentiments only)")
            print(f"   Sentiment distribution:")
            for sent_idx in sentiment_order:
                if sent_idx in unique:
                    idx_in_unique = np.where(unique == sent_idx)[0][0]
                    count = counts[idx_in_unique]
                else:
                    count = 1  # Avoid division by zero
                
                sentiment_name = train_dataset.idx_to_sentiment[sent_idx]
                # For "none", show percentage of total_all; for others, show percentage of total
                if sent_idx == 3:
                    pct = (count / total_all * 100) if total_all > 0 else 0
                    print(f"     {sentiment_name:10s}: {count:6,} ({pct:5.2f}% of all)")
                else:
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"     {sentiment_name:10s}: {count:6,} ({pct:5.2f}%)")
                
                # Inverse frequency: total / (num_classes * count)
                # For "none" class (idx=3), always use minimal weight
                if sent_idx == 3:  # none
                    weight = 0.01  # Fixed minimal weight (not used in loss anyway)
                else:
                    weight = total / (num_classes * count)
                
                alpha.append(weight)
            
            print(f"\n   Calculated alpha (inverse frequency):")
            for sent_idx, weight in zip(sentiment_order, alpha):
                sentiment_name = train_dataset.idx_to_sentiment[sent_idx]
                print(f"     {sentiment_name:10s}: {weight:.4f}")
        
        elif isinstance(focal_alpha_config, list) and len(focal_alpha_config) == 4:
            print(f"\nAlpha mode: USER-DEFINED (global)")
            alpha = focal_alpha_config
            print(f"   Using custom alpha: {alpha}")
        
        elif focal_alpha_config is None:
            print(f"\nAlpha mode: EQUAL (no class weighting)")
            alpha = [1.0, 1.0, 1.0, 1.0]  # negative, neutral, positive, none
            print(f"   Using equal weights: {alpha}")
        
        else:
            print(f"\n[WARNING]️  Invalid focal_alpha config: {focal_alpha_config}")
            print(f"   Falling back to AUTO mode")
            alpha = calculate_global_alpha(
                config['paths']['train_file'],
                train_dataset.aspects,
                sentiment_to_idx
            )
        
        # Create Focal Loss
        focal_loss_fn = MultilabelFocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            num_aspects=11
        )
        focal_loss_fn = focal_loss_fn.to(device)
        
        print(f"\n[OK] Focal Loss ready:")
        print(f"   Gamma: {focal_gamma}")
        print(f"   Alpha: {alpha}")
        print(f"   Mode: Global (same alpha for all 11 aspects)")
    
    # Create model
    print(f"\nCreating model...")
    model = MultiLabelPhoBERT(
        model_name=config['model']['name'],
        num_aspects=11,
        num_sentiments=4,  # Negative, Neutral, Positive, None
        hidden_size=512,
        dropout=0.3
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    # Use epochs from config unless explicitly overridden
    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"\n[WARNING] Using epochs from command line: {num_epochs} (overrides config: {config['training'].get('num_train_epochs')})")
    else:
        num_epochs = config['training'].get('num_train_epochs', 5)
        print(f"\n[OK] Using epochs from config: {num_epochs}")
    
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n[OK] Training setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}")
    
    best_f1 = 0.0
    aspect_names = train_dataset.aspects
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, focal_loss_fn)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names)
        print_metrics(val_metrics)
        
        # Save checkpoint
        is_best = val_metrics['overall_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['overall_f1']
            print(f"\n[NEW BEST] F1: {best_f1*100:.2f}%")
        
        # Use output_dir from config if not specified
        output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            output_dir, is_best=is_best
        )
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")
    
    # Use output_dir from config if not specified
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    
    # Load best checkpoint
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, aspect_names)
    print_metrics(test_metrics)
    
    # Save final results
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect']
    }
    
    import json
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[OK] Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\n✅ Best Model Performance:")
    print(f"   Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   Test F1:       {test_metrics['overall_f1']*100:.2f}%")
    print(f"\nModel saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config if specified)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config if specified)')
    
    args = parser.parse_args()
    main(args)
