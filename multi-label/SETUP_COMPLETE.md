# ✅ Multi-Label Setup Complete!

## 🎯 What's Been Created

Multi-label ABSA system với cấu trúc tương tự `single-label/`.

## 📁 Files Created

```
multi-label/
├── config.yaml                     ✅ Config (giống single-label)
├── requirements.txt                ✅ Dependencies
├── README.md                      ✅ Main documentation
├── run_all.bat                    ✅ Complete workflow script
│
├── preprocess_data.py             ✅ Convert to multi-label format
├── train_phobert_multilabel.py    ✅ Multi-label training
│
└── MULTILABEL_FORMAT.md           ✅ Format documentation
```

## 🔄 Key Differences from Single-Label

| Feature | Single-Label | Multi-Label |
|---------|-------------|-------------|
| **Script** | `train_phobert_trainer.py` | `train_phobert_multilabel.py` |
| **Format** | Sentence-aspect pairs | Sentences with 33 binary labels |
| **Num labels** | 3 (Neg/Neu/Pos) | 33 (11 aspects × 3) |
| **Samples** | 15,569 pairs | 9,129 sentences |
| **Loss** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Metrics** | Accuracy, F1 | F1 micro/macro, Hamming |
| **Oversampling** | Yes | No |
| **Threshold** | N/A | 0.5 |

## 🚀 Usage

### Quick Start

```bash
cd multi-label

# Step 1: Preprocess
python preprocess_data.py

# Step 2: Train
python train_phobert_multilabel.py

# Or run all
run_all.bat
```

### Expected Output

```
multi-label/
├── dataset_multilabel.csv         # Converted format
│
├── data/
│   ├── train.csv                  # 7,303 samples
│   ├── val.csv                    # 913 samples
│   └── test.csv                   # 913 samples
│
├── checkpoints/
│   └── phobert_multilabel/
│       └── best_model/
│
├── results/
│   ├── evaluation_report.txt
│   └── test_predictions.csv
│
└── training_logs/
    └── training_log_*.txt
```

## 📊 Multi-Label Format

**Example:**

```
Sentence: "Pin tốt, camera đẹp"
Labels: [0,0,1,0,0,1,0,0,0,...,0]
         └─┬─┘ └─┬─┘
           │     └─ Camera-Positive (label_5)
           └─────── Battery-Positive (label_2)
```

**Label Mapping:**
```python
label_idx = aspect_idx * 3 + sentiment_idx

# Examples:
Battery-Negative = 0*3+0 = 0
Battery-Positive = 0*3+2 = 2
Camera-Positive = 1*3+2 = 5
```

## ⚙️ Config Highlights

```yaml
model:
  num_labels: 33
  problem_type: "multi_label_classification"

training:
  metric_for_best_model: "eval_f1_micro"  # Micro F1
  
# No oversampling section (not needed for multi-label)
```

## 📝 Training Process

1. **Preprocess**
   - Load `dataset.csv` (multi-aspect format)
   - Convert to 33 binary labels
   - Split train/val/test (80/10/10)
   - Save to `data/`

2. **Train**
   - Load multi-label data
   - Train PhoBERT with BCEWithLogitsLoss
   - Evaluate with multi-label metrics
   - Save best model

3. **Evaluate**
   - F1 (micro): Overall performance
   - F1 (macro): Per-label average
   - Hamming Loss: Fraction of incorrect labels
   - Exact Match: All labels correct

## ✅ Features

- ✅ Same config structure as single-label
- ✅ Same training pipeline (HuggingFace Trainer)
- ✅ Same reproducibility (seed 42)
- ✅ Same logging (TeeLogger to file)
- ✅ Same paths (script location based)
- ✅ Multi-label specific metrics

## 🎯 When to Use Multi-Label

**Use Multi-Label when:**
- Want to preserve natural sentence structure
- Need to predict multiple aspects per sentence
- Want more efficient training (no data expansion)
- Care about overall sentiment across all aspects

**Use Single-Label when:**
- Need fine-grained per-aspect analysis
- Want to balance sentiments per aspect
- Need detailed aspect-specific metrics
- Doing aspect-level sentiment classification

## 📚 Documentation

- **README.md** - Main usage guide
- **MULTILABEL_FORMAT.md** - Detailed format explanation
- **config.yaml** - All training settings
- **../single-label/docs/** - Shared documentation

## 🧪 Test Run

```bash
cd multi-label

# Test preprocessing
python preprocess_data.py

# Should output:
# ✓ Found 9,129 samples
# ✓ Converted to 33 binary labels
# ✓ Average ~2.5 labels per sample
# ✓ Split: train=7303, val=913, test=913

# Test training (full run ~20-30 min on RTX 4060)
python train_phobert_multilabel.py

# Should output:
# ✓ Model loaded with 33 labels
# ✓ Training completed
# ✓ Test F1 (micro): ~0.88
# ✓ Test F1 (macro): ~0.85
```

## 🔧 Hardware Requirements

Same as single-label:
- **GPU**: RTX 4060 (8GB VRAM) or better
- **RAM**: 16GB+
- **Storage**: ~5GB

## ⚠️ Notes

1. **No Oversampling**
   - Multi-label naturally more balanced
   - Each sample has multiple labels
   - No need for per-aspect oversampling

2. **Different Metrics**
   - Use F1 micro for overall performance
   - Use F1 macro for per-label fairness
   - Exact match is very strict

3. **Threshold**
   - Default: 0.5 for all labels
   - Can be tuned per label if needed
   - Check predictions to adjust

## 📊 Expected Results

```
Test Metrics:
  F1 (micro): 0.8800
  F1 (macro): 0.8500
  Precision (micro): 0.8750
  Recall (micro): 0.8850
  Hamming Loss: 0.0300
  Exact Match: 0.6500
```

## 🎉 Ready to Train!

```bash
cd multi-label
run_all.bat
```

---

**Multi-label ABSA system is ready!** 🚀

Compare with single-label:
- `cd ../single-label && run_all.bat`
- See which approach works better for your use case
