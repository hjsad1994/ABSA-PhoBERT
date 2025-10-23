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
    Convert multi-aspect dataset - keep same format with aspect columns
    
    Input format (dataset.csv):
        data, Battery, Camera, ..., Others (11 aspects with Negative/Neutral/Positive)
    
    Output format:
        sentence, Battery, Camera, ..., Others (11 aspects with Negative/Neutral/Positive)
    """
    logger.info(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    logger.info(f"Found {len(df)} samples")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Get aspect columns
    aspects = config['valid_aspects']
    
    # Validate columns
    missing_aspects = [asp for asp in aspects if asp not in df.columns]
    if missing_aspects:
        raise ValueError(f"Missing aspect columns: {missing_aspects}")
    
    # Create output dataframe
    result_df = pd.DataFrame()
    
    # Use 'data' column from input, rename to 'sentence' for output
    text_column = 'data' if 'data' in df.columns else 'sentence'
    result_df['sentence'] = df[text_column]
    
    # Copy aspect columns (keep Negative/Neutral/Positive text values)
    for aspect in aspects:
        result_df[aspect] = df[aspect]
    
    # Calculate statistics
    total_labels = 0
    sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
    aspect_counts = {aspect: 0 for aspect in aspects}
    
    logger.info("Processing aspect sentiments...")
    for idx, row in result_df.iterrows():
        for aspect in aspects:
            sentiment = row[aspect]
            if pd.notna(sentiment) and sentiment in ['Negative', 'Neutral', 'Positive']:
                total_labels += 1
                sentiment_counts[sentiment] += 1
                aspect_counts[aspect] += 1
    
    avg_labels_per_sample = total_labels / len(result_df)
    
    logger.info(f"Converted {len(result_df)} samples")
    logger.info(f"Total aspect-sentiment pairs: {total_labels}")
    logger.info(f"Average aspects per sample: {avg_labels_per_sample:.2f}")
    
    # Show sentiment distribution
    logger.info("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        logger.info(f"  {sentiment}: {count} ({count/total_labels*100:.1f}%)")
    
    # Show aspect distribution (top 10)
    logger.info("\nAspect distribution (top 10):")
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for aspect, count in sorted_aspects:
        logger.info(f"  {aspect}: {count} mentions")
    
    # Save
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved to {output_file}")
    
    return result_df


def split_data(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split data into train/val/test sets with stratification"""
    logger.info(f"\nSplitting data into train/val/test...")
    logger.info(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {random_state}")
    logger.info(f"Using stratified split based on dominant sentiment")
    
    df = pd.read_csv(input_file)
    
    # Get aspect columns from config
    config = load_config()
    aspects = config['valid_aspects']
    
    # Create stratification label based on dominant sentiment
    def get_dominant_sentiment(row):
        """Find the most common sentiment across all aspects"""
        sentiments = []
        for aspect in aspects:
            if pd.notna(row[aspect]) and row[aspect] in ['Negative', 'Neutral', 'Positive']:
                sentiments.append(row[aspect])
        
        if not sentiments:
            return 'None'  # No sentiment
        
        # Count sentiments and return most common
        from collections import Counter
        counter = Counter(sentiments)
        return counter.most_common(1)[0][0]
    
    logger.info("Creating stratification labels based on dominant sentiment...")
    df['_stratify'] = df.apply(get_dominant_sentiment, axis=1)
    
    logger.info(f"Stratification distribution:")
    stratify_dist = df['_stratify'].value_counts()
    for label, count in stratify_dist.items():
        logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Remove classes with too few samples for stratification
    # Need at least ceil(1 / val_ratio) samples to ensure >= 2 in val/test split
    min_samples_needed = max(10, int(1 / min(val_ratio, test_ratio)) + 2)
    classes_to_keep = stratify_dist[stratify_dist >= min_samples_needed].index
    df_stratified = df[df['_stratify'].isin(classes_to_keep)].copy()
    
    if len(df_stratified) < len(df):
        removed = len(df) - len(df_stratified)
        logger.info(f"Note: Removed {removed} samples with rare stratification labels (< {min_samples_needed} samples)")
        logger.info(f"  Keeping {len(df_stratified)} samples for stratified split")
    
    # First split: train vs (val+test) with stratification
    train_df, temp_df = train_test_split(
        df_stratified,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df_stratified['_stratify']
    )
    
    # Second split: val vs test with stratification
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        stratify=temp_df['_stratify']
    )
    
    # Remove stratification column before saving
    train_df = train_df.drop('_stratify', axis=1)
    val_df = val_df.drop('_stratify', axis=1)
    test_df = test_df.drop('_stratify', axis=1)
    
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
    config = load_config()
    aspects = config['valid_aspects']
    
    logger.info(f"\nLabel statistics:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        total_labels = 0
        for aspect in aspects:
            total_labels += split_df[aspect].notna().sum()
        avg_labels = total_labels / len(split_df)
        logger.info(f"  {name}: {int(total_labels)} total aspect-sentiments, {avg_labels:.2f} avg per sample")


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
