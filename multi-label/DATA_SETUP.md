# Data Setup for Multi-Label Training

## Issue

The `multi-label/` directory doesn't have a `data/` folder. You need to prepare training data first.

---

## Current Data Location

Training data exists in:
```
single-label/data/
├── train.csv
├── train_oversampled.csv
├── test.csv
└── validation.csv (or val.csv)
```

---

## Option 1: Copy Data from single-label (Recommended)

```bash
# From project root (E:\ABSA-PhoBERT\)
mkdir multi-label\data
copy single-label\data\*.csv multi-label\data\
```

Or in PowerShell:
```powershell
New-Item -ItemType Directory -Path "multi-label\data" -Force
Copy-Item "single-label\data\*.csv" -Destination "multi-label\data\"
```

---

## Option 2: Update Config to Use Existing Data

Edit `multi-label/config.yaml`:

```yaml
paths:
  # Point to single-label data folder
  train_file: "../single-label/data/train_oversampled.csv"
  validation_file: "../single-label/data/val.csv"  
  test_file: "../single-label/data/test.csv"
```

---

## Data Format

### Current Format: Binary Labels

Your data uses **binary labels** (0/1 for each of 33 labels):

```csv
sentence,label_0,label_1,label_2,...,label_32
"đúng với hình ảnh",0,0,0,...,1,0,0
"Chất lượng tốt",0,0,0,...,1,0,0
```

**Label structure:**
- 11 aspects × 3 sentiments = 33 binary labels
- Each aspect has 3 labels: [Negative, Neutral, Positive]
  - Battery: label_0 (neg), label_1 (neu), label_2 (pos)
  - Camera: label_3 (neg), label_4 (neu), label_5 (pos)
  - ...etc

---

## Dataset Class Features

The `MultiLabelABSADataset` class automatically converts binary format to sentiment labels:

### Conversion Logic:

```python
# Binary format (CSV):
Battery_Negative=1, Battery_Neutral=0, Battery_Positive=0

# Converted to:
Battery -> sentiment_idx=1 (negative)

# Sentiment mapping:
positive -> 0
negative -> 1  
neutral -> 2
```

### Priority Rules:

When multiple sentiments are active:
1. **Negative** has highest priority
2. **Neutral** has medium priority
3. **Positive** has lowest priority

When no sentiment is active:
- Default to **neutral** (idx=2)

---

## Alternative: Text Format (Optional)

If you want to use text labels instead of binary:

### CSV Format:

```csv
sentence,Battery,Camera,Performance,...,Others
"Pin tốt",positive,neutral,neutral,...,neutral
"Camera xấu",neutral,negative,neutral,...,neutral
```

### Usage:

```python
dataset = MultiLabelABSADataset(
    'data/train_text.csv',
    tokenizer,
    max_length=256,
    format_type='text'  # Instead of 'binary'
)
```

---

## Verify Data Setup

After copying/setting up data, test it:

```bash
cd multi-label
python dataset_multilabel.py
```

Expected output:
```
================================================================================
Testing MultiLabelABSADataset
================================================================================

1. Testing binary format...
Loaded 7303 samples from ...train.csv
Format: binary
Aspects: 11

[OK] Dataset loaded: 7303 samples

First sample:
  Input IDs shape: torch.Size([256])
  Attention mask shape: torch.Size([256])
  Labels shape: torch.Size([11])
  Labels: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

Sentiment distribution across all aspects:
  positive  :  4,523 ( 56.32%)
  negative  :  2,145 ( 26.71%)
  neutral   :  1,365 ( 16.99%)

[OK] Binary format test passed!
```

---

## Training After Setup

Once data is ready:

```bash
# Train with default config
python train_multilabel.py --config config.yaml

# Or specify data location
python train_multilabel.py \
  --config config.yaml \
  --epochs 15
```

---

## Data Requirements

### Minimum Requirements:

- **Training samples**: 1,000+ recommended
- **Validation samples**: 100+ recommended  
- **Test samples**: 100+ recommended

### Format Requirements:

**Binary format (current):**
- Columns: `sentence`, `label_0` to `label_32` (34 total)
- Values: 0 or 1 for each label

**Text format (alternative):**
- Columns: `sentence`, aspect names (11 total)
- Values: `positive`, `negative`, `neutral`, or empty

---

## Troubleshooting

### Issue: "No such file or directory: 'data/train.csv'"

**Solution:** Data folder doesn't exist. Use Option 1 or 2 above.

### Issue: "Missing binary label columns"

**Solution:** CSV must have all 33 labels (label_0 to label_32).

Check your CSV:
```python
import pandas as pd
df = pd.read_csv('data/train.csv')
print(df.columns.tolist())
```

Should see: `['sentence', 'label_0', 'label_1', ..., 'label_32']`

### Issue: "Unknown sentiment 'xxx'"

**Solution:** If using text format, sentiment values must be:
- `positive` (not `pos`, `Positive`, etc.)
- `negative` (not `neg`, `Negative`, etc.)
- `neutral` (not `neu`, `Neutral`, etc.)

Clean your data:
```python
df['Battery'] = df['Battery'].str.lower().str.strip()
```

---

## Quick Start Commands

### 1. Copy data:
```bash
mkdir multi-label\data
copy single-label\data\*.csv multi-label\data\
```

### 2. Test dataset:
```bash
cd multi-label
python dataset_multilabel.py
```

### 3. Train model:
```bash
python train_multilabel.py --config config.yaml --epochs 15
```

---

## Summary

- ✅ Created `dataset_multilabel.py` - handles both binary and text formats
- ⚠️ Need to setup `data/` folder - use copy command above
- ✅ Sentiment mapping: positive=0, negative=1, neutral=2
- ✅ Ready to train with PhoBERT + Focal Loss once data is ready

**Next step:** Copy data files, then run training!
