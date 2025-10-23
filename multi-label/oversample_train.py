"""
Oversampling for Multi-Label ABSA
Oversample samples with rare labels to balance label distribution
"""
import os
import yaml
import pandas as pd
import numpy as np
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_label_frequencies(df, num_labels=33):
    """Calculate frequency of each label"""
    label_cols = [f'label_{i}' for i in range(num_labels)]
    frequencies = df[label_cols].sum().values
    return frequencies


def oversample_multilabel(df, num_labels=33, target_min_samples=500, random_state=42):
    """
    Oversample samples with rare labels
    
    Strategy:
    1. Calculate frequency of each label
    2. Identify rare labels (below target_min_samples)
    3. Oversample samples containing rare labels
    
    Args:
        df: DataFrame with sentence and label_0...label_32
        num_labels: Number of labels (default: 33)
        target_min_samples: Target minimum samples per label
        random_state: Random seed
    
    Returns:
        Oversampled DataFrame
    """
    logger.info("=" * 60)
    logger.info("Multi-Label Oversampling")
    logger.info("=" * 60)
    logger.info(f"Random seed: {random_state}")
    logger.info(f"Target min samples per label: {target_min_samples}")
    
    np.random.seed(random_state)
    
    label_cols = [f'label_{i}' for i in range(num_labels)]
    
    # Calculate initial label frequencies
    initial_freq = calculate_label_frequencies(df, num_labels)
    logger.info(f"\nInitial dataset: {len(df)} samples")
    logger.info(f"Label frequency range: {int(initial_freq.min())} - {int(initial_freq.max())}")
    
    # Find rare labels
    rare_labels = np.where(initial_freq < target_min_samples)[0]
    logger.info(f"\nRare labels (< {target_min_samples} samples): {len(rare_labels)}")
    
    if len(rare_labels) == 0:
        logger.info("No rare labels found. Skipping oversampling.")
        return df
    
    # Log rare labels
    logger.info("\nRare labels:")
    for label_idx in rare_labels[:10]:  # Show top 10
        count = int(initial_freq[label_idx])
        logger.info(f"  label_{label_idx}: {count} samples (need {target_min_samples - count} more)")
    
    # Oversample
    oversampled_dfs = [df]  # Start with original data
    
    for label_idx in rare_labels:
        label_col = f'label_{label_idx}'
        current_count = int(initial_freq[label_idx])
        needed_samples = target_min_samples - current_count
        
        if needed_samples <= 0:
            continue
        
        # Get samples with this label
        samples_with_label = df[df[label_col] == 1]
        
        if len(samples_with_label) == 0:
            logger.warning(f"No samples found for {label_col}, skipping")
            continue
        
        # Sample with replacement
        oversampled = samples_with_label.sample(
            n=needed_samples, 
            replace=True, 
            random_state=random_state + label_idx
        )
        
        oversampled_dfs.append(oversampled)
    
    # Combine all
    result_df = pd.concat(oversampled_dfs, ignore_index=True)
    
    # Shuffle
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate final frequencies
    final_freq = calculate_label_frequencies(result_df, num_labels)
    
    logger.info(f"\n" + "=" * 60)
    logger.info("Oversampling Results")
    logger.info("=" * 60)
    logger.info(f"Original: {len(df)} samples")
    logger.info(f"Oversampled: {len(result_df)} samples")
    logger.info(f"Increase: +{len(result_df) - len(df)} samples ({(len(result_df)/len(df) - 1)*100:.1f}%)")
    logger.info(f"\nLabel frequency range:")
    logger.info(f"  Before: {int(initial_freq.min())} - {int(initial_freq.max())}")
    logger.info(f"  After:  {int(final_freq.min())} - {int(final_freq.max())}")
    
    # Show improvement for rare labels
    logger.info("\nImprovement for rare labels (top 10):")
    for label_idx in rare_labels[:10]:
        before = int(initial_freq[label_idx])
        after = int(final_freq[label_idx])
        improvement = after - before
        logger.info(f"  label_{label_idx}: {before} → {after} (+{improvement})")
    
    return result_df


def main():
    logger.info("=" * 60)
    logger.info("Multi-Label Oversampling for ABSA")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Get seed
    random_seed = config.get('reproducibility', {}).get('oversampling_seed', 42)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input/output paths
    data_dir = os.path.join(script_dir, 'data')
    input_file = os.path.join(data_dir, 'train.csv')
    output_file = os.path.join(data_dir, 'train_oversampled.csv')
    
    # Check if input exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run preprocess_data.py first!")
        return
    
    logger.info(f"\nInput: {input_file}")
    logger.info(f"Output: {output_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Oversample
    oversampled_df = oversample_multilabel(
        df,
        num_labels=config['model']['num_labels'],
        target_min_samples=500,  # Target: at least 500 samples per label
        random_state=random_seed
    )
    
    # Save
    oversampled_df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info(f"\n✓ Oversampled data saved to: {output_file}")
    logger.info("")


if __name__ == '__main__':
    main()
