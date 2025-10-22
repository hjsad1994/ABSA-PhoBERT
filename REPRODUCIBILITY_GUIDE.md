# Reproducibility Guide for Research

## 🎯 Mục Đích

Đảm bảo kết quả nghiên cứu có thể **tái tạo (reproducible)** với cùng kết quả khi chạy lại với cùng seed.

## 🔧 Configuration

### Seed Configuration trong `config.yaml`

```yaml
reproducibility:
  # Master seed - applied to all random operations
  seed: 42
  
  # Data preprocessing
  data_split_seed: 42        # Seed for train/val/test split
  oversampling_seed: 42      # Seed for per-aspect oversampling
  shuffle_seed: 42           # Seed for data shuffling
  
  # Training
  training_seed: 42          # Seed for model training
  data_loader_seed: 42       # Seed for data loader workers
```

**Quan trọng:** Tất cả seeds phải giống nhau (42) để đảm bảo reproducibility!

## 📊 Workflow

### 1. Data Preprocessing (với seed)

```bash
# Sử dụng seed từ config
python preprocess_data.py

# Hoặc override seed từ command line
python preprocess_data.py --seed 42
```

**Output:**
- `data/train.csv` (12,455 samples)
- `data/val.csv` (1,556 samples)
- `data/test.csv` (1,558 samples)

**Reproducible:** Chạy lại với cùng seed sẽ cho cùng train/val/test split!

### 2. Oversampling (với seed)

```bash
# Sử dụng seed từ config
python oversample_train.py

# Hoặc override seed từ command line
python oversample_train.py --seed 42
```

**Output:**
- `data/train_oversampled.csv` (21,060 samples)

**Reproducible:** Chạy lại với cùng seed sẽ cho cùng oversampled data!

### 3. Training (với seed)

```bash
# Seed được đọc từ config tự động
python train_phobert_trainer.py
```

**Seed được sử dụng:**
- Model initialization
- Weight initialization
- Dropout
- Data shuffling trong DataLoader
- Optimizer state

**Reproducible:** Chạy lại với cùng seed sẽ cho cùng training trajectory!

## ✅ Verification

### Kiểm tra cấu hình:

```bash
python verify_reproducibility.py
```

**Script này sẽ check:**
1. ✓ All seeds trong config có giống nhau không
2. ✓ Data split có reproducible không
3. ✓ Oversampling có reproducible không
4. ✓ Files đã tồn tại chưa

**Output mẫu:**

```
================================================================================
REPRODUCIBILITY VERIFICATION FOR RESEARCH
================================================================================

================================================================================
SEED CONFIGURATION CHECK
================================================================================

Master Seed: 42

Seed Configuration:
  ✓ Data Split         : 42
  ✓ Oversampling       : 42
  ✓ Shuffle            : 42
  ✓ Training           : 42
  ✓ Data Loader        : 42

✅ All seeds are consistent!
   All random operations use seed: 42

================================================================================
DATA SPLIT REPRODUCIBILITY TEST
================================================================================

Testing with seed: 42

✅ Data split is reproducible!
   Same seed produces identical train/val/test splits

================================================================================
OVERSAMPLING REPRODUCIBILITY TEST
================================================================================

Testing with seed: 42

✅ Oversampling is reproducible!
   Same seed produces identical oversampled datasets

================================================================================
SUMMARY
================================================================================

✅ ALL CHECKS PASSED!

Your configuration ensures reproducible results:
  • All seeds are consistent (seed=42)
  • Data splitting is reproducible
  • Oversampling is reproducible
```

## 🔬 Multiple Experiments (Different Seeds)

Để nghiên cứu robust, chạy experiments với nhiều seeds khác nhau:

### Experiment 1: Seed 42
```bash
# Update config.yaml: seed: 42
python preprocess_data.py --seed 42
python oversample_train.py --seed 42
python train_phobert_trainer.py
# Rename output: mv checkpoints/phobert_finetuned checkpoints/phobert_seed42
```

### Experiment 2: Seed 123
```bash
# Update config.yaml: seed: 123
python preprocess_data.py --seed 123
python oversample_train.py --seed 123
python train_phobert_trainer.py
# Rename output: mv checkpoints/phobert_finetuned checkpoints/phobert_seed123
```

### Experiment 3: Seed 456
```bash
# Update config.yaml: seed: 456
python preprocess_data.py --seed 456
python oversample_train.py --seed 456
python train_phobert_trainer.py
# Rename output: mv checkpoints/phobert_finetuned checkpoints/phobert_seed456
```

### So sánh kết quả:

```python
import pandas as pd
import matplotlib.pyplot as plt

results = {
    'seed_42': {
        'f1': 0.9234,
        'accuracy': 0.9312
    },
    'seed_123': {
        'f1': 0.9198,
        'accuracy': 0.9287
    },
    'seed_456': {
        'f1': 0.9256,
        'accuracy': 0.9328
    }
}

# Report mean ± std
f1_scores = [r['f1'] for r in results.values()]
print(f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
# Example: F1: 0.9229 ± 0.0024
```

## 📝 Best Practices for Research

### 1. ✅ Document Your Seeds
```yaml
# config.yaml
reproducibility:
  seed: 42  # Main experiment
  # Alternative seeds for robustness: 123, 456, 789, 2024
```

### 2. ✅ Version Control Your Data Splits
```bash
# After running preprocess_data.py with seed 42
git add data/train.csv data/val.csv data/test.csv
git commit -m "Data split with seed 42"
```

### 3. ✅ Save Seed Info with Results
```python
# In your paper/report
"""
Results (seed=42):
  F1 Score: 0.9234
  Accuracy: 0.9312
  
Results averaged over 3 seeds (42, 123, 456):
  F1 Score: 0.9229 ± 0.0024
  Accuracy: 0.9309 ± 0.0017
"""
```

### 4. ✅ Log Seeds in Training Logs
```
Training log sẽ tự động ghi:
  Random seed for training: 42
  (Ensures reproducible results for research)
```

## 🚫 Common Mistakes

### ❌ Mistake 1: Hardcoded Seeds
```python
# BAD - hardcoded
df.sample(frac=1, random_state=42)

# GOOD - từ config
seed = config['reproducibility']['seed']
df.sample(frac=1, random_state=seed)
```

### ❌ Mistake 2: Inconsistent Seeds
```yaml
# BAD
reproducibility:
  data_split_seed: 42
  training_seed: 123  # Different!

# GOOD
reproducibility:
  data_split_seed: 42
  training_seed: 42   # Same!
```

### ❌ Mistake 3: Không kiểm tra Reproducibility
```bash
# ALWAYS verify before experiments
python verify_reproducibility.py
```

## 📚 Files Modified for Reproducibility

1. **config.yaml**
   - Added `reproducibility` section with all seeds

2. **preprocess_data.py**
   - Reads `data_split_seed` from config
   - Accepts `--seed` argument to override
   - Logs seed being used

3. **oversample_train.py**
   - Reads `oversampling_seed` from config
   - Accepts `--seed` argument to override
   - Logs seed being used

4. **train_phobert_trainer.py**
   - Reads `training_seed` from config
   - Priority: `reproducibility.training_seed` > `training.seed` > `general.seed`
   - Logs seed being used

5. **verify_reproducibility.py** (NEW)
   - Verification script
   - Tests data split reproducibility
   - Tests oversampling reproducibility

## ✅ Checklist

Trước khi chạy experiments:

- [ ] Đã cập nhật `config.yaml` với seeds nhất quán
- [ ] Đã chạy `python verify_reproducibility.py` và pass tất cả checks
- [ ] Đã document seed đang sử dụng
- [ ] Đã version control data splits (nếu dùng git)
- [ ] Đã backup kết quả với seed info

## 🎓 For Academic Papers

### Reporting Reproducibility:

```latex
\section{Experimental Setup}

All experiments were conducted with fixed random seeds to ensure 
reproducibility. We used seed 42 for data splitting (80/10/10 
train/validation/test), per-aspect oversampling, and model training. 
To evaluate robustness, we also ran experiments with seeds 123, 456, 
789, and 2024. Results are reported as mean ± standard deviation 
across all five seeds.

\subsection{Reproducibility}

Our code and configuration are available at [repository link]. 
To reproduce our results:

1. Install dependencies: \texttt{pip install -r requirements.txt}
2. Preprocess data: \texttt{python preprocess\_data.py --seed 42}
3. Oversample training data: \texttt{python oversample\_train.py --seed 42}
4. Train model: \texttt{python train\_phobert\_trainer.py}

All random operations use the same seed (42) for full reproducibility.
```

---

**✅ Reproducibility đã được đảm bảo! Sẵn sàng cho nghiên cứu!**
