# Global Alpha Calculation Results

## Overview

Calculated global alpha weights for multi-label ABSA using **inverse frequency** formula.

```python
alpha = total_active_labels / (num_classes * count_per_class)
```

This gives **higher weight to rare classes** and **lower weight to common classes**.

---

## Training Data Statistics

```
Dataset: data/train_oversampled.csv
Samples: 12,790
Total active labels: 22,752 (across 11 aspects × 3 sentiments)
Average labels per sample: 1.78
```

---

## Global Sentiment Distribution

Counting all sentiments across ALL 11 aspects:

| Sentiment | Count | Percentage | Rarity |
|-----------|-------|------------|--------|
| **Negative** | 7,348 | 32.30% | Slightly rare |
| **Neutral**  | 7,102 | 31.21% | **Most rare** |
| **Positive** | 8,302 | 36.49% | Most common |

---

## Calculated Alpha Weights

Using inverse frequency formula: `alpha = total / (3 * count)`

| Sentiment | Alpha | Effect |
|-----------|-------|--------|
| **Negative** | **1.0321** | Boost by 3.2% (slightly rare) |
| **Neutral**  | **1.0679** | **Boost by 6.8%** (most rare) |
| **Positive** | **0.9135** | Reduce by 8.7% (most common) |

**Interpretation:**
- Neutral is rarest (31.2%) → Gets highest weight (1.0679)
- Positive is most common (36.5%) → Gets lowest weight (0.9135)
- Weights are **balanced** around 1.0 (data is fairly balanced)

---

## Usage in Focal Loss

### For Your MultilabelFocalLoss Class:

```python
from focal_loss import MultilabelFocalLoss

# Use calculated global alpha
focal_loss = MultilabelFocalLoss(
    alpha=[1.0321, 1.0679, 0.9135],  # [Negative, Neutral, Positive]
    gamma=2.0,
    num_aspects=11
)

# Forward pass
loss = focal_loss(logits, labels)
```

### Config.yaml:

```yaml
training:
  loss_type: "focal"
  focal_alpha: [1.0321, 1.0679, 0.9135]  # [Negative, Neutral, Positive]
  focal_gamma: 2.0
```

---

## Comparison: Auto vs Manual Alpha

### Previous Approach (Binary Classification)

```yaml
focal_alpha: 0.2322  # Single value for binary positive/negative
```

- Treats each of 33 labels independently
- Alpha controls positive vs negative balance
- Calculated from overall positive frequency (~5.4% → sqrt → 0.23)

### Current Approach (3-Class Weights)

```yaml
focal_alpha: [1.0321, 1.0679, 0.9135]  # Three values for Neg/Neu/Pos
```

- Treats data as multi-class (3 sentiments)
- Alpha controls weight of each sentiment class
- Calculated from global sentiment distribution (Inverse frequency)

---

## Expected Impact

### Why These Alphas Help:

1. **Neutral Boost (1.0679)**:
   - Neutral is rarest (31.2%)
   - Gets 6.8% more weight
   - Model pays more attention to neutral samples
   - Should improve neutral class recall

2. **Positive Reduction (0.9135)**:
   - Positive is most common (36.5%)
   - Gets 8.7% less weight
   - Prevents model from being biased toward positive
   - Balances precision/recall across all classes

3. **Overall Balance**:
   - All alphas are close to 1.0
   - Data is relatively balanced
   - Focal loss will primarily help with **hard examples** (via gamma)
   - Alpha provides fine-tuning for class imbalance

---

## Formula Details

### Inverse Frequency Formula:

```python
alpha[class_i] = total_samples / (num_classes * count[class_i])
```

**Example for Neutral:**
```
total = 22,752 active labels
num_classes = 3
count_neutral = 7,102

alpha_neutral = 22,752 / (3 × 7,102)
              = 22,752 / 21,306
              = 1.0679
```

### Why This Works:

- **Rare class** (low count) → high alpha → more weight
- **Common class** (high count) → low alpha → less weight
- Automatic balancing based on data distribution
- No manual tuning needed

---

## Re-calculate for Different Data

To recalculate for your own training data:

```bash
cd multi-label
python calculate_alpha.py
```

This will:
1. Load training data
2. Count global sentiment distribution
3. Calculate inverse frequency alpha
4. Print results and config

---

## Next Steps

1. **Implement** your `MultilabelFocalLoss` class with these alpha values
2. **Train** your model with focal loss
3. **Compare** with baseline (BCE) and previous focal loss (single alpha)
4. **Evaluate** impact on neutral class performance

Expected improvements:
- Better neutral class recall (+3-5%)
- More balanced F1 scores across all sentiments
- Overall F1 improvement (+1-2%)

---

**Generated:** 2025-10-23  
**Training Data:** train_oversampled.csv (12,790 samples)  
**Formula:** Inverse Frequency (global across all aspects)
