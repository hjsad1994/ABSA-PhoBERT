"""
Per-Aspect Oversampling Script for Training Data
Balances sentiment labels independently within each aspect
Input: data/train.csv
Output: data/train_oversampled.csv
"""
import os
import yaml
import pandas as pd
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def oversample_per_aspect(df, aspect_col='aspect', label_col='sentiment', seed=42):
    """
    Oversample data per aspect: balance labels within each aspect
    
    Args:
        df: DataFrame with sentence-aspect-sentiment data
        aspect_col: Name of aspect column
        label_col: Name of sentiment label column
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with balanced labels per aspect
    """
    logger.info("=" * 80)
    logger.info("Per-Aspect Oversampling")
    logger.info("=" * 80)
    
    unique_aspects = df[aspect_col].unique()
    oversampled_dfs = []
    
    logger.info(f"Found {len(unique_aspects)} aspects")
    logger.info("")
    
    total_before = len(df)
    
    for aspect in sorted(unique_aspects):
        aspect_df = df[df[aspect_col] == aspect].copy()
        
        # Count labels for this aspect
        label_counts = aspect_df[label_col].value_counts()
        max_count = label_counts.max()
        
        logger.info(f"Aspect: {aspect}")
        logger.info(f"  Before: {dict(label_counts.sort_index())}")
        logger.info(f"  Target: {max_count} samples per label")
        
        aspect_samples = []
        
        for label in label_counts.index:
            label_samples = aspect_df[aspect_df[label_col] == label]
            count = len(label_samples)
            
            if count < max_count:
                # Oversample
                n_samples_needed = max_count - count
                oversampled = label_samples.sample(n=n_samples_needed, replace=True, random_state=seed)
                combined = pd.concat([label_samples, oversampled], ignore_index=True)
                aspect_samples.append(combined)
                logger.info(f"  '{label}': {count} → {len(combined)} (+{n_samples_needed})")
            else:
                aspect_samples.append(label_samples)
        
        aspect_balanced = pd.concat(aspect_samples, ignore_index=True)
        oversampled_dfs.append(aspect_balanced)
        logger.info(f"  After: {len(aspect_balanced)} samples")
        logger.info("")
    
    # Combine all aspects and shuffle
    result_df = pd.concat(oversampled_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    total_after = len(result_df)
    increase_pct = ((total_after - total_before) / total_before) * 100
    
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total samples: {total_before} → {total_after} (+{total_after - total_before}, +{increase_pct:.1f}%)")
    
    # Show overall label distribution
    logger.info("")
    logger.info("Overall distribution before:")
    for label, count in df[label_col].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    logger.info("")
    logger.info("Overall distribution after:")
    for label, count in result_df[label_col].value_counts().sort_index().items():
        pct = (count / len(result_df)) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    return result_df


def main():
    """Main function to oversample training data"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Per-aspect oversampling for ABSA training data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config if provided)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PhoBERT ABSA - Training Data Oversampling")
    print("=" * 80)
    print("")
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Get seed from config or argument
    if args.seed is not None:
        random_seed = args.seed
        logger.info(f"Using seed from argument: {random_seed}")
    else:
        random_seed = config.get('reproducibility', {}).get('oversampling_seed', 42)
        logger.info(f"Using seed from config: {random_seed}")
    
    logger.info(f"Random seed for oversampling: {random_seed}")
    logger.info("")
    
    # File paths
    input_file = "data/train.csv"
    output_file = "data/train_oversampled.csv"
    
    # Check input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run preprocess_data.py first to create train/val/test split")
        return
    
    # Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check required columns
    required_cols = ['sentence', 'aspect', 'sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Found columns: {df.columns.tolist()}")
        return
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("")
    
    # Apply oversampling
    df_oversampled = oversample_per_aspect(df, seed=random_seed)
    
    # Save result
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Saving oversampled data to: {output_file}")
    df_oversampled.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved {len(df_oversampled)} samples")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration used:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Random seed: {random_seed}")
    logger.info("")
    logger.info("Done! Use 'train_oversampled.csv' for training.")


if __name__ == '__main__':
    main()
