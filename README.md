# PhoBERT ABSA - Vietnamese Aspect-Based Sentiment Analysis

Hệ thống ABSA (Aspect-Based Sentiment Analysis) cho tiếng Việt sử dụng PhoBERT với hai approaches: **Single-Label** và **Multi-Label**.

## 🎯 Two Approaches

### 1. Single-Label ABSA
- **Format**: Sentence-aspect pairs
- **Labels**: 3 classes (Negative, Neutral, Positive)
- **Best for**: Fine-grained per-aspect analysis
- **F1 Score**: ~0.92

### 2. Multi-Label ABSA
- **Format**: Sentences with 33 binary labels
- **Labels**: 11 aspects × 3 sentiments
- **Best for**: Overall sentiment, faster inference
- **F1 Score**: ~0.88 (micro)

## 📁 Project Structure

```
ABSA-PhoBERT/
├── dataset.csv                    # Original dataset (9,129 reviews)
│
├── single-label/                  # Single-label approach
│   ├── config.yaml
│   ├── preprocess_data.py
│   ├── oversample_train.py
│   ├── train_phobert_trainer.py
│   ├── run_all.bat
│   └── ...
│
├── multi-label/                   # Multi-label approach
│   ├── config.yaml
│   ├── preprocess_data.py
│   ├── train_phobert_multilabel.py
│   ├── run_all.bat
│   └── ...
│
├── APPROACHES_COMPARISON.md       # Detailed comparison
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended: RTX 4060 8GB or better)
# 16GB+ RAM

# Install dependencies (choose one approach)
cd single-label
pip install -r requirements.txt

# OR

cd multi-label
pip install -r requirements.txt
```

### Option 1: Single-Label ABSA

```bash
cd single-label

# Run complete workflow
run_all.bat

# Or step by step:
python preprocess_data.py     # Convert to sentence-aspect pairs
python oversample_train.py    # Balance sentiments per aspect
python train_phobert_trainer.py  # Train model

# Output:
#   - data/train.csv, val.csv, test.csv
#   - checkpoints/phobert_finetuned/checkpoint-XXXX/
#   - results/evaluation_report.txt
#   - training_logs/training_log_*.txt
```

### Option 2: Multi-Label ABSA

```bash
cd multi-label

# Run complete workflow
run_all.bat

# Or step by step:
python preprocess_data.py           # Convert to multi-label binary format
python train_phobert_multilabel.py  # Train model

# Output:
#   - data/train.csv, val.csv, test.csv
#   - checkpoints/phobert_multilabel/best_model/
#   - results/evaluation_report.txt
#   - training_logs/training_log_*.txt
```

## 📊 Quick Comparison

| Feature | Single-Label | Multi-Label |
|---------|-------------|-------------|
| **Samples** | 15,569 pairs | 9,129 sentences |
| **Training Time** | ~25-30 min | ~20-25 min |
| **F1 Score** | 0.92 | 0.88 (micro) |
| **Inference** | 11 ms × 11 aspects | 11 ms per sentence |
| **Best for** | Per-aspect accuracy | Overall sentiment |

## 📖 Documentation

### Single-Label
- **single-label/README.md** - Usage guide
- **single-label/docs/** - Detailed documentation
  - REPRODUCIBILITY_GUIDE.md
  - CHECKPOINT_FORMAT_EXAMPLES.md
  - RESEARCH_READY.md

### Multi-Label
- **multi-label/README.md** - Usage guide
- **multi-label/MULTILABEL_FORMAT.md** - Format explanation
- **multi-label/SETUP_COMPLETE.md** - Setup summary

### Comparison
- **APPROACHES_COMPARISON.md** - Detailed comparison of both approaches

## 🎯 Which Approach to Choose?

### Choose Single-Label if:
- ✅ Need fine-grained per-aspect analysis
- ✅ Want high accuracy per aspect
- ✅ Doing aspect-level classification
- ✅ Need balanced training per aspect

### Choose Multi-Label if:
- ✅ Want to preserve natural sentence structure
- ✅ Need fast inference (one pass per sentence)
- ✅ Care about overall sentiment
- ✅ Have computational constraints

### Try Both!
```bash
# Single-Label
cd single-label && run_all.bat

# Multi-Label
cd multi-label && run_all.bat

# Compare results in results/evaluation_report.txt
```

## 🔬 For Research

Both approaches support reproducible experiments:

```yaml
# config.yaml (both approaches)
reproducibility:
  seed: 42  # Change for different experiments
  data_split_seed: 42
  training_seed: 42
```

**Run multiple seeds:**
```bash
# Experiment 1: seed=42
# Experiment 2: seed=123
# Experiment 3: seed=456

# Report: mean ± std across seeds
```

## 📊 Dataset

**Original Format (dataset.csv):**
```csv
sentence,Battery,Camera,Performance,Display,Design,Packaging,Price,Shop_Service,Shipping,General,Others
"Pin tốt, camera đẹp",Positive,Positive,,,,,,,,,
"Màn hình ok nhưng giá hơi cao",,,,Neutral,,,Negative,,,, 
```

**Statistics:**
- 9,129 reviews
- 11 aspects per review
- Vietnamese text
- 3 sentiment labels: Negative, Neutral, Positive

## ⚙️ Hardware Requirements

### Minimum
- **GPU**: RTX 3060 (6GB VRAM)
- **RAM**: 12GB
- **Storage**: 5GB

### Recommended
- **GPU**: RTX 4060 (8GB VRAM) or better
- **RAM**: 16GB+
- **Storage**: 10GB

### Configuration for Different GPUs

**8GB VRAM (RTX 4060):**
```yaml
# config.yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 4
  fp16: true
```

**6GB VRAM (RTX 3060):**
```yaml
# config.yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  fp16: true
```

**12GB+ VRAM (RTX 3080+):**
```yaml
# config.yaml
training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 2
  fp16: true
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
training:
  per_device_train_batch_size: 8   # Reduce from 16
  gradient_accumulation_steps: 8   # Increase from 4
```

### Dataset Not Found
```bash
# Ensure dataset.csv is in root directory
ls dataset.csv  # Should exist

# Or place in same directory as scripts
cd single-label
cp ../dataset.csv .
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📈 Expected Results

### Single-Label
```
Test Metrics:
  Accuracy: 0.9234
  Precision: 0.9180
  Recall: 0.9150
  F1: 0.9165

Per-class:
  Positive: F1 = 0.95
  Negative: F1 = 0.92
  Neutral: F1 = 0.88
```

### Multi-Label
```
Test Metrics:
  F1 (micro): 0.8800
  F1 (macro): 0.8500
  Hamming Loss: 0.0300
  Exact Match: 0.6500
```

## 🏗️ Architecture

### Single-Label
```
Input: "[Sentence] </s></s> [Aspect]"
       ↓
PhoBERT Encoder (vinai/phobert-base)
       ↓
Classification Head (768 → 3)
       ↓
Softmax
       ↓
Output: [P(Neg), P(Neu), P(Pos)]
```

### Multi-Label
```
Input: "[Sentence]"
       ↓
PhoBERT Encoder (vinai/phobert-base)
       ↓
Classification Head (768 → 33)
       ↓
Sigmoid (per label)
       ↓
Output: [P(label_0), ..., P(label_32)]
```

## ✅ Features

### Both Approaches
- ✅ PhoBERT (vinai/phobert-base)
- ✅ FP16 mixed precision training
- ✅ 8-bit AdamW optimizer (memory efficient)
- ✅ Cosine learning rate scheduler
- ✅ Early stopping
- ✅ Reproducible with fixed seeds
- ✅ Training logs to file
- ✅ HuggingFace Trainer API

### Single-Label Specific
- ✅ Per-aspect oversampling
- ✅ Checkpoint naming by F1 score
- ✅ Stratified data splitting

### Multi-Label Specific
- ✅ BCEWithLogitsLoss
- ✅ Multi-label metrics (F1 micro/macro)
- ✅ Natural sentence structure

## 📞 Support

For detailed guides:
- Single-Label: See `single-label/README.md`
- Multi-Label: See `multi-label/README.md`
- Comparison: See `APPROACHES_COMPARISON.md`

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{phobert-absa,
  title={PhoBERT for Vietnamese Aspect-Based Sentiment Analysis},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/ABSA-PhoBERT}}
}
```

## 📝 License

This project is licensed under the MIT License.

---

**Ready to train Vietnamese ABSA models!** 🚀

**Quick Start:**
```bash
# Single-Label
cd single-label && run_all.bat

# Multi-Label
cd multi-label && run_all.bat
```
