# Multi-Label ABSA with PhoBERT

Hệ thống Aspect-Based Sentiment Analysis (ABSA) sử dụng PhoBERT với multi-label approach (mỗi câu có thể có nhiều aspect-sentiment pairs).

## 📁 Cấu Trúc Thư Mục

```
multi-label/
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── run_all.bat                    # Run complete workflow
│
├── preprocess_data.py             # Data preprocessing
├── train_phobert_multilabel.py    # Multi-label training script
│
├── data/                          # Created by preprocess_data.py
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── checkpoints/                   # Created by training
│   └── phobert_multilabel/
│
├── results/                       # Created by training
│   ├── evaluation_report.txt
│   └── test_predictions.csv
│
└── training_logs/                 # Created by training
    └── training_log_*.txt
```

## 🎯 Multi-Label Format

**Khác biệt với Single-Label:**

### Single-Label:
- Mỗi sentence-aspect pair → 1 sentiment
- Format: `sentence, aspect, sentiment`
- Example: "Pin tốt", "Battery", "Positive"

### Multi-Label:
- Mỗi sentence → nhiều aspect-sentiment pairs
- Format: `sentence, label_0, label_1, ..., label_32` (33 binary labels)
- Example: "Pin tốt, camera đẹp" → Battery-Positive=1, Camera-Positive=1, others=0

**Label Mapping:**
```
label_idx = aspect_idx * 3 + sentiment_idx

Battery:      Negative=0, Neutral=1, Positive=2
Camera:       Negative=3, Neutral=4, Positive=5
Performance:  Negative=6, Neutral=7, Positive=8
Display:      Negative=9, Neutral=10, Positive=11
Design:       Negative=12, Neutral=13, Positive=14
Packaging:    Negative=15, Neutral=16, Positive=17
Price:        Negative=18, Neutral=19, Positive=20
Shop_Service: Negative=21, Neutral=22, Positive=23
Shipping:     Negative=24, Neutral=25, Positive=26
General:      Negative=27, Neutral=28, Positive=29
Others:       Negative=30, Neutral=31, Positive=32
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd multi-label
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
# Convert multi-aspect to multi-label binary format
python preprocess_data.py

# Output:
#   - dataset_multilabel.csv (converted format)
#   - data/train.csv (7,303 samples)
#   - data/val.csv (913 samples)
#   - data/test.csv (913 samples)
```

### 3. Train Model

```bash
# Train PhoBERT with multi-label classification
python train_phobert_multilabel.py

# Output:
#   - checkpoints/phobert_multilabel/
#   - results/evaluation_report.txt
#   - training_logs/training_log_YYYYMMDD_HHMMSS.txt
```

### 4. Complete Workflow

```bash
run_all.bat
```

## ⚙️ Configuration (config.yaml)

### Key Settings

```yaml
model:
  name: "vinai/phobert-base"
  num_labels: 33  # 11 aspects × 3 sentiments
  problem_type: "multi_label_classification"

training:
  per_device_train_batch_size: 16
  num_train_epochs: 5
  learning_rate: 2.0e-5
  metric_for_best_model: "eval_f1_micro"  # Micro F1 for multi-label

# No oversampling needed (multi-label is naturally balanced)
```

## 📊 Multi-Label Metrics

### Micro Average
- Treats each label prediction equally
- Good for overall performance
- **F1 Micro**: Harmonic mean of micro precision/recall

### Macro Average
- Average metrics across all labels
- Each label has equal weight
- Good for per-label performance

### Other Metrics
- **Hamming Loss**: Fraction of incorrect labels
- **Exact Match**: Percentage of samples with all labels correct

## 🔬 For Research

### Reproducibility

All random operations use the same seed (42):

```yaml
reproducibility:
  seed: 42
  data_split_seed: 42
  training_seed: 42
  data_loader_seed: 42
```

### Expected Performance

- **F1 (micro)**: ~0.88
- **F1 (macro)**: ~0.85
- **Exact Match**: ~0.65
- **Hamming Loss**: ~0.03

## 📝 Workflow

```
dataset.csv (multi-aspect)
         ↓
preprocess_data.py (convert to multi-label)
         ↓
dataset_multilabel.csv (33 binary labels)
         ↓
train/val/test split
         ↓
train_phobert_multilabel.py
         ↓
checkpoints/ + results/ + training_logs/
```

## ✅ Features

- ✅ PhoBERT (vinai/phobert-base)
- ✅ Multi-label classification (BCEWithLogitsLoss)
- ✅ 8-bit AdamW optimizer (memory efficient)
- ✅ FP16 mixed precision training
- ✅ Multi-label metrics (micro/macro F1)
- ✅ Reproducible with fixed seeds
- ✅ HuggingFace Trainer API

## 🎯 Hardware Requirements

- **GPU**: RTX 4060 (8GB VRAM) or better
- **RAM**: 16GB+
- **Storage**: ~5GB for model + data

## 📊 Dataset Statistics

### Original (dataset.csv)
- 9,129 reviews
- 11 aspects per review
- Multi-aspect format

### After Preprocessing (dataset_multilabel.csv)
- **9,129 samples** (same as original)
- **33 binary labels** per sample
- **Average ~2-3 active labels** per sample
- No oversampling needed

### Splits
- **Train**: 7,303 samples (80%)
- **Val**: 913 samples (10%)
- **Test**: 913 samples (10%)

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# config.yaml
training:
  per_device_train_batch_size: 8   # Reduce from 16
  gradient_accumulation_steps: 8   # Increase from 4
```

### Poor Performance
- Check label distribution (some labels might be very rare)
- Try different thresholds (default: 0.5)
- Increase training epochs

## 🆚 Single-Label vs Multi-Label

| Aspect | Single-Label | Multi-Label |
|--------|-------------|-------------|
| **Format** | sentence-aspect pairs | sentences with binary labels |
| **Samples** | 15,569 pairs | 9,129 sentences |
| **Labels** | 3 (Negative/Neutral/Positive) | 33 (11×3 binary) |
| **Loss** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Metrics** | Accuracy, F1 | F1 micro/macro, Hamming |
| **Oversampling** | Yes (per-aspect) | No (naturally balanced) |
| **Best for** | Fine-grained analysis | Overall sentiment |

## 📞 Support

Check `../single-label/docs/` for detailed guides on:
- Reproducibility
- Training configuration
- Troubleshooting

---

**Multi-label ABSA system ready for training!** 🚀
