import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def convert_multi_aspect_to_single_label(input_file: str, output_file: str):
    """
    Convert multi-aspect dataset to single-label format.
    
    Input format: data, Battery, Camera, Performance, Display, Design, Packaging, Price, Shop_Service, Shipping, General, Others
    Output format: sentence, aspect, sentiment
    
    Each row with multiple aspects is expanded into multiple rows (one per non-empty aspect).
    """
    logger.info(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Define aspect columns
    aspect_columns = [
        'Battery', 'Camera', 'Performance', 'Display', 'Design', 
        'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others'
    ]
    
    # Text column
    text_column = 'data'
    
    # Verify columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    
    missing_aspects = [col for col in aspect_columns if col not in df.columns]
    if missing_aspects:
        logger.warning(f"Missing aspect columns: {missing_aspects}")
        aspect_columns = [col for col in aspect_columns if col in df.columns]
    
    logger.info(f"Found {len(df)} reviews with {len(aspect_columns)} aspects")
    
    # Expand dataset: one row per text-aspect pair
    expanded_data = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        # Skip if text is empty or NaN
        if pd.isna(text) or str(text).strip() == '':
            continue
        
        for aspect in aspect_columns:
            label = row[aspect]
            
            # Skip if label is empty or NaN
            if pd.isna(label) or str(label).strip() == '':
                continue
            
            # Clean label (remove whitespace)
            label = str(label).strip()
            
            # Validate label
            if label not in ['Positive', 'Negative', 'Neutral']:
                logger.warning(f"Invalid label '{label}' for aspect '{aspect}' at row {idx}, skipping")
                continue
            
            expanded_data.append({
                'sentence': text,
                'aspect': aspect,
                'sentiment': label
            })
    
    # Create output dataframe
    output_df = pd.DataFrame(expanded_data)
    
    # Log statistics
    logger.info(f"Expanded to {len(output_df)} sentence-aspect pairs")
    logger.info(f"\nSentiment distribution:")
    logger.info(output_df['sentiment'].value_counts())
    logger.info(f"\nAspect distribution:")
    logger.info(output_df['aspect'].value_counts())
    
    # Save to file
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved expanded dataset to {output_file}")
    
    return output_df


def split_data(input_file: str, output_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        input_file: path to the single-label CSV file
        output_dir: directory to save train/val/test files
        train_ratio: proportion for training set (default: 0.8)
        val_ratio: proportion for validation set (default: 0.1)
        test_ratio: proportion for test set (default: 0.1)
        random_state: random seed for reproducibility
    """
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Verify ratios sum to 1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split data
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_file = output_path / 'train.csv'
    val_file = output_path / 'val.csv'
    test_file = output_path / 'test.csv'
    
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    val_df.to_csv(val_file, index=False, encoding='utf-8')
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    
    # Log statistics
    logger.info(f"\nData split complete:")
    logger.info(f"Train: {len(train_df)} samples ({train_ratio*100:.1f}%) -> {train_file}")
    logger.info(f"Val:   {len(val_df)} samples ({val_ratio*100:.1f}%) -> {val_file}")
    logger.info(f"Test:  {len(test_df)} samples ({test_ratio*100:.1f}%) -> {test_file}")
    
    # Log sentiment distribution per split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        logger.info(f"\n{split_name} sentiment distribution:")
        logger.info(split_df['sentiment'].value_counts())


def main():
    """Main preprocessing pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocess ABSA dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config if provided)')
    args = parser.parse_args()
    
    # Load config
    logger.info("=" * 60)
    logger.info("Loading Configuration")
    logger.info("=" * 60)
    config = load_config(args.config)
    
    # Get seed from config or argument
    if args.seed is not None:
        random_seed = args.seed
        logger.info(f"Using seed from argument: {random_seed}")
    else:
        random_seed = config.get('reproducibility', {}).get('data_split_seed', 42)
        logger.info(f"Using seed from config: {random_seed}")
    
    logger.info(f"Random seed for all operations: {random_seed}")
    
    # Step 1: Convert multi-aspect to single-label format
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Converting multi-aspect dataset to single-label format")
    logger.info("=" * 60)
    
    input_file = 'dataset.csv'
    expanded_file = 'dataset_expanded.csv'
    
    convert_multi_aspect_to_single_label(input_file, expanded_file)
    
    # Step 2: Split into train/val/test
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Splitting data into train/val/test sets")
    logger.info("=" * 60)
    
    output_dir = 'data'
    split_data(
        expanded_file, 
        output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=random_seed  # Use seed from config
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"Configuration used:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Random seed: {random_seed}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review the generated files in the 'data/' directory")
    logger.info(f"2. (Optional) Run oversampling: python oversample_train.py")
    logger.info(f"3. Run training: python train_phobert_trainer.py")


if __name__ == '__main__':
    main()
