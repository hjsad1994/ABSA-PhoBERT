"""
Verify Per-Aspect Oversampling
Checks if each aspect has balanced sentiment labels
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_per_aspect_balance(csv_file):
    """Verify that each aspect has balanced sentiment labels"""
    logger.info("=" * 80)
    logger.info(f"Verifying Per-Aspect Balance: {csv_file}")
    logger.info("=" * 80)
    logger.info("")
    
    # Load data
    df = pd.read_csv(csv_file)
    logger.info(f"Total samples: {len(df)}")
    logger.info("")
    
    # Check each aspect
    aspects = sorted(df['aspect'].unique())
    all_balanced = True
    
    for aspect in aspects:
        aspect_df = df[df['aspect'] == aspect]
        sentiment_counts = aspect_df['sentiment'].value_counts().sort_index()
        
        logger.info(f"Aspect: {aspect}")
        logger.info(f"  Total: {len(aspect_df)} samples")
        
        # Show distribution
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(aspect_df)) * 100
            logger.info(f"    {sentiment}: {count} ({pct:.1f}%)")
        
        # Check if balanced (all counts should be equal)
        unique_counts = sentiment_counts.unique()
        
        if len(unique_counts) == 1:
            logger.info(f"  ✓ BALANCED - All sentiments have {unique_counts[0]} samples")
        else:
            logger.info(f"  ✗ IMBALANCED - Counts: {sentiment_counts.tolist()}")
            all_balanced = False
        
        logger.info("")
    
    # Overall summary
    logger.info("=" * 80)
    logger.info("Overall Distribution")
    logger.info("=" * 80)
    
    overall = df['sentiment'].value_counts().sort_index()
    for sentiment, count in overall.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    logger.info("")
    logger.info("=" * 80)
    
    if all_balanced:
        logger.info("✓ ALL ASPECTS ARE BALANCED!")
    else:
        logger.info("✗ SOME ASPECTS ARE IMBALANCED")
    
    logger.info("=" * 80)
    
    return all_balanced


def main():
    """Main verification function"""
    print("\n" + "=" * 80)
    print("PhoBERT ABSA - Oversampling Verification")
    print("=" * 80 + "\n")
    
    # Verify original file
    print("1. Original Training Data (Before Oversampling)")
    print("-" * 80)
    verify_per_aspect_balance("data/train.csv")
    
    print("\n\n")
    
    # Verify oversampled file
    print("2. Oversampled Training Data (After Oversampling)")
    print("-" * 80)
    verify_per_aspect_balance("data/train_oversampled.csv")


if __name__ == '__main__':
    main()
