"""
Data Preprocessing for Multi-Label ABSA
Converts multi-aspect dataset to multi-label binary format
"""
import os
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def convert_to_multilabel(input_file, output_file, config):
    """
    Convert multi-aspect dataset to multi-label binary format
    
    Input format (dataset.csv):
        sentence, Battery, Camera, ..., Others (11 aspects)
    
    Output format:
        sentence, label_0, label_1, ..., label_32 (33 binary labels)
        
    Label mapping:
        label_idx = aspect_idx * 3 + sentiment_idx
        - Battery_Negative = 0, Battery_Neutral = 1, Battery_Positive = 2
        - Camera_Negative = 3, Camera_Neutral = 4, Camera_Positive = 5
        - ...
    """
    logger.info(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    logger.info(f"Found {len(df)} samples")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Get aspect columns
    aspects = config['valid_aspects']
    sentiment_map = config['sentiment_labels']
    
    # Validate columns
    missing_aspects = [asp for asp in aspects if asp not in df.columns]
    if missing_aspects:
        raise ValueError(f"Missing aspect columns: {missing_aspects}")
    
    # Create output dataframe
    result_df = pd.DataFrame()
    # Use 'data' column from input, rename to 'sentence' for output
    text_column = 'data' if 'data' in df.columns else 'sentence'
    result_df['sentence'] = df[text_column]
    
    # Create 33 binary labels
    num_labels = len(aspects) * 3
    label_columns = [f'label_{i}' for i in range(num_labels)]
    
    # Initialize all labels to 0
    for col in label_columns:
        result_df[col] = 0
    
    # Convert sentiments to binary labels
    logger.info("Converting to multi-label format...")
    for idx, row in df.iterrows():
        for aspect_idx, aspect in enumerate(aspects):
            sentiment = row[aspect]
            
            # Skip if no sentiment or invalid
            if pd.isna(sentiment) or sentiment not in sentiment_map:
                continue
            
            # Calculate label index
            sentiment_idx = sentiment_map[sentiment]
            label_idx = aspect_idx * 3 + sentiment_idx
            
            # Set binary label
            result_df.at[idx, f'label_{label_idx}'] = 1
    
    # Calculate statistics
    total_labels = result_df[label_columns].sum().sum()
    avg_labels_per_sample = total_labels / len(result_df)
    
    logger.info(f"Converted {len(result_df)} samples")
    logger.info(f"Total active labels: {int(total_labels)}")
    logger.info(f"Average labels per sample: {avg_labels_per_sample:.2f}")
    
    # Show label distribution
    label_counts = result_df[label_columns].sum()
    logger.info("\nLabel distribution (top 10):")
    top_labels = label_counts.nlargest(10)
    for label, count in top_labels.items():
        label_idx = int(label.split('_')[1])
        aspect_idx = label_idx // 3
        sentiment_idx = label_idx % 3
        aspect_name = aspects[aspect_idx]
        sentiment_name = [k for k, v in sentiment_map.items() if v == sentiment_idx][0]
        logger.info(f"  {label} ({aspect_name}-{sentiment_name}): {int(count)} samples")
    
    # Save
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved to {output_file}")
    
    return result_df


def split_data(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split data into train/val/test sets"""
    logger.info(f"\nSplitting data into train/val/test...")
    logger.info(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {random_state}")
    
    df = pd.read_csv(input_file)
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state
    )
    
    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    logger.info(f"\nSplit complete:")
    logger.info(f"  Train: {len(train_df)} samples → {train_path}")
    logger.info(f"  Val:   {len(val_df)} samples → {val_path}")
    logger.info(f"  Test:  {len(test_df)} samples → {test_path}")
    
    # Calculate label statistics for each split
    label_cols = [col for col in df.columns if col.startswith('label_')]
    
    logger.info(f"\nLabel statistics:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        total_labels = split_df[label_cols].sum().sum()
        avg_labels = total_labels / len(split_df)
        logger.info(f"  {name}: {int(total_labels)} total labels, {avg_labels:.2f} avg per sample")


def main():
    logger.info("=" * 60)
    logger.info("Multi-Label ABSA Data Preprocessing")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Get seed
    random_seed = config.get('reproducibility', {}).get('data_split_seed', 42)
    logger.info(f"Random seed: {random_seed}")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find dataset.csv
    if os.path.exists('dataset.csv'):
        input_file = 'dataset.csv'
    elif os.path.exists('../dataset.csv'):
        input_file = '../dataset.csv'
    else:
        logger.error("dataset.csv not found in current or parent directory!")
        logger.error(f"Current directory: {os.getcwd()}")
        return
    
    logger.info(f"Found dataset at: {input_file}")
    
    # Output paths (in multi-label/ directory)
    multilabel_file = os.path.join(script_dir, 'dataset_multilabel.csv')
    output_dir = os.path.join(script_dir, 'data')
    
    # Step 1: Convert to multi-label format
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Converting to multi-label format")
    logger.info("=" * 60)
    
    convert_to_multilabel(input_file, multilabel_file, config)
    
    # Step 2: Split data
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Splitting data")
    logger.info("=" * 60)
    
    split_data(
        multilabel_file,
        output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=random_seed
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - train.csv")
    logger.info(f"  - val.csv")
    logger.info(f"  - test.csv")


if __name__ == '__main__':
    main()
