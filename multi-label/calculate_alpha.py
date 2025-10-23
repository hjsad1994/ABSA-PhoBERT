"""
Calculate Global Alpha Weights for Focal Loss
==============================================
Computes inverse frequency alpha for [Negative, Neutral, Positive] sentiments
"""

import pandas as pd
import numpy as np


def calculate_global_alpha(train_file, num_aspects=11):
    """
    Calculate global alpha weights from binary label data
    
    Args:
        train_file: Path to training CSV with binary labels
        num_aspects: Number of aspects (default: 11)
    
    Returns:
        alpha: [alpha_negative, alpha_neutral, alpha_positive]
    """
    print(f"\n{'='*80}")
    print("Calculating Global Alpha Weights (Inverse Frequency)")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(train_file, encoding='utf-8')
    
    # Labels format: label_0, label_1, ..., label_32
    # Every 3 labels = 1 aspect: [Negative, Neutral, Positive]
    # label_i where i = aspect_idx * 3 + sentiment_idx
    
    num_sentiments = 3
    total_labels = num_aspects * num_sentiments
    
    # Count each sentiment across ALL aspects
    count_negative = 0
    count_neutral = 0
    count_positive = 0
    
    for aspect_idx in range(num_aspects):
        # Get indices for this aspect's 3 sentiments
        neg_idx = aspect_idx * 3 + 0  # Negative
        neu_idx = aspect_idx * 3 + 1  # Neutral  
        pos_idx = aspect_idx * 3 + 2  # Positive
        
        # Count active labels (1s)
        count_negative += df[f'label_{neg_idx}'].sum()
        count_neutral += df[f'label_{neu_idx}'].sum()
        count_positive += df[f'label_{pos_idx}'].sum()
    
    # Total active labels
    total_active = count_negative + count_neutral + count_positive
    
    print(f"Training data: {len(df):,} samples")
    print(f"Total active labels: {int(total_active):,}\n")
    
    print("Sentiment distribution (global across all aspects):")
    print(f"  {'Sentiment':<12} {'Count':<10} {'Percentage':<12}")
    print(f"  {'-'*40}")
    
    for sentiment, count in [('Negative', count_negative), 
                              ('Neutral', count_neutral), 
                              ('Positive', count_positive)]:
        pct = (count / total_active * 100) if total_active > 0 else 0
        print(f"  {sentiment:<12} {int(count):<10,} {pct:>6.2f}%")
    
    # Calculate alpha using inverse frequency formula
    # alpha = total / (num_classes * count)
    # This gives more weight to rare classes
    
    alpha_negative = total_active / (num_sentiments * max(count_negative, 1))
    alpha_neutral = total_active / (num_sentiments * max(count_neutral, 1))
    alpha_positive = total_active / (num_sentiments * max(count_positive, 1))
    
    print(f"\n{'='*80}")
    print("Calculated Alpha (Inverse Frequency)")
    print(f"{'='*80}")
    print(f"  Alpha[Negative]: {alpha_negative:.4f}")
    print(f"  Alpha[Neutral]:  {alpha_neutral:.4f}")
    print(f"  Alpha[Positive]: {alpha_positive:.4f}")
    print(f"{'='*80}\n")
    
    print("Interpretation:")
    if alpha_negative > 1.0:
        print(f"  - Negative is RARE ({count_negative/total_active*100:.1f}%) -> Higher weight ({alpha_negative:.2f}x)")
    elif alpha_negative < 1.0:
        print(f"  - Negative is COMMON ({count_negative/total_active*100:.1f}%) -> Lower weight ({alpha_negative:.2f}x)")
    else:
        print(f"  - Negative is BALANCED ({count_negative/total_active*100:.1f}%) -> Normal weight (1.00x)")
    
    if alpha_neutral > 1.0:
        print(f"  - Neutral is RARE ({count_neutral/total_active*100:.1f}%) -> Higher weight ({alpha_neutral:.2f}x)")
    elif alpha_neutral < 1.0:
        print(f"  - Neutral is COMMON ({count_neutral/total_active*100:.1f}%) -> Lower weight ({alpha_neutral:.2f}x)")
    else:
        print(f"  - Neutral is BALANCED ({count_neutral/total_active*100:.1f}%) -> Normal weight (1.00x)")
    
    if alpha_positive > 1.0:
        print(f"  - Positive is RARE ({count_positive/total_active*100:.1f}%) -> Higher weight ({alpha_positive:.2f}x)")
    elif alpha_positive < 1.0:
        print(f"  - Positive is COMMON ({count_positive/total_active*100:.1f}%) -> Lower weight ({alpha_positive:.2f}x)")
    else:
        print(f"  - Positive is BALANCED ({count_positive/total_active*100:.1f}%) -> Normal weight (1.00x)")
    
    print(f"\n{'='*80}")
    print("Config Update")
    print(f"{'='*80}")
    print("Add this to your config.yaml:")
    print(f"""
training:
  loss_type: "focal"
  focal_alpha: [{alpha_negative:.4f}, {alpha_neutral:.4f}, {alpha_positive:.4f}]  # [Negative, Neutral, Positive]
  focal_gamma: 2.0
""")
    
    return [alpha_negative, alpha_neutral, alpha_positive]


if __name__ == '__main__':
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, 'data', 'train_oversampled.csv')
    
    if not os.path.exists(train_file):
        train_file = os.path.join(script_dir, 'data', 'train.csv')
    
    alpha = calculate_global_alpha(train_file)
    
    print(f"\nDone! Use these alpha values in your focal loss:\n")
    print(f"   alpha = {alpha}")
