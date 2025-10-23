# Per-Aspect Oversampling Workflow

## Overview

The oversampling process is now separated into a standalone preprocessing step, making it cleaner and more modular.

## Files

- **oversample_train.py** - Standalone script for per-aspect oversampling
- **data/train.csv** - Original training data (12,455 samples)
- **data/train_oversampled.csv** - Balanced training data (21,060 samples)

## Oversampling Statistics

### Original Distribution
- **Total samples**: 12,455
- **Negative**: 5,115 (41.1%)
- **Neutral**: 1,645 (13.2%)
- **Positive**: 5,695 (45.7%)

### After Oversampling
- **Total samples**: 21,060 (+69.1%)
- **Negative**: 6,965 (33.1%)
- **Neutral**: 7,130 (33.9%)
- **Positive**: 6,965 (33.1%)

### Per-Aspect Balancing

Each of the 11 aspects is balanced independently:

| Aspect | Original | Oversampled | Increase |
|--------|----------|-------------|----------|
| Battery | 1,414 | 2,637 | +86.5% |
| Camera | 958 | 1,674 | +74.7% |
| Design | 1,158 | 2,391 | +106.5% |
| Display | 716 | 1,011 | +41.2% |
| General | 2,047 | 2,724 | +33.1% |
| Others | 165 | 165 | 0% (already balanced) |
| Packaging | 1,026 | 1,581 | +54.1% |
| Performance | 1,176 | 1,797 | +52.8% |
| Price | 1,127 | 2,715 | +140.9% |
| Shipping | 1,446 | 2,415 | +67.0% |
| Shop_Service | 1,222 | 1,950 | +59.6% |

## Workflow

### 1. Generate Oversampled Training Data

Run the oversampling script once after creating the train/val/test split:

```bash
python oversample_train.py
```

This will:
- Read `data/train.csv`
- Balance labels within each aspect
- Save to `data/train_oversampled.csv`
- Log detailed statistics

### 2. Configure Training

In `config.yaml`, set:

```yaml
data:
  use_oversampled_file: true  # Use pre-generated oversampled file
```

Options:
- `use_oversampled_file: true` → Load `train_oversampled.csv` (no runtime oversampling)
- `use_oversampled_file: false` → Load `train.csv` and apply oversampling on-the-fly

### 3. Train Model

```bash
python train_phobert_trainer.py
```

The training script will:
- Check the config flag
- Load the appropriate file
- Fall back to on-the-fly oversampling if oversampled file is missing

## Benefits of This Approach

1. **Separation of Concerns**: Data preprocessing is separate from training
2. **Reproducibility**: Same oversampled data across multiple training runs
3. **Flexibility**: Can easily switch between original and oversampled data
4. **Performance**: No overhead of oversampling during each training session
5. **Debugging**: Can inspect the oversampled data directly

## Implementation Details

### Oversampling Strategy

For each aspect:
1. Find the majority class count (max samples)
2. For each minority class:
   - Randomly sample with replacement to match the majority count
   - Append oversampled data to original
3. Shuffle all samples with fixed seed (42) for reproducibility

### Label Distribution

The per-aspect oversampling ensures:
- **Within each aspect**: All sentiment labels are perfectly balanced
- **Overall**: More balanced distribution (~33% each vs. 41%/13%/46%)
- **No data loss**: Original samples are preserved, only duplicates are added

## Next Steps

After oversampling is complete:
1. ✅ Run training with oversampled data
2. ✅ Compare performance with/without oversampling
3. ✅ Monitor GPU usage and training time
4. ✅ Evaluate on test set (no oversampling on test data!)
