# Checkpoint Naming Format - F1 Score (4 Digits)

## 🎯 Format Mới

Checkpoint được đặt tên theo **F1 score với 4 chữ số** (2 số thập phân)

### Công thức:
```
F1 Score × 10000 = Checkpoint Name
```

### Ví dụ:

| F1 Score | Phần trăm | Checkpoint Name | Tên File |
|----------|-----------|-----------------|----------|
| 0.8723 | 87.23% | 8723 | `checkpoint-8723` |
| 0.8753 | 87.53% | 8753 | `checkpoint-8753` |
| 0.9145 | 91.45% | 9145 | `checkpoint-9145` |
| 0.9234 | 92.34% | 9234 | `checkpoint-9234` |
| 0.9567 | 95.67% | 9567 | `checkpoint-9567` |

## 📊 Output Mẫu Khi Training

### Console Output:
```
Epoch 1/5: 100%|████████| 389/389 [05:23<00:00, 1.20it/s]
Evaluation: 100%|████████| 25/25 [00:15<00:00, 1.65it/s]

📁 Renamed: checkpoint-389 -> checkpoint-8723 (eval_f1=87.23%)

Epoch 2/5: 100%|████████| 389/389 [05:20<00:00, 1.21it/s]
Evaluation: 100%|████████| 25/25 [00:14<00:00, 1.72it/s]

📁 Renamed: checkpoint-778 -> checkpoint-9145 (eval_f1=91.45%)

Epoch 3/5: 100%|████████| 389/389 [05:18<00:00, 1.22it/s]
Evaluation: 100%|████████| 25/25 [00:14<00:00, 1.78it/s]

📁 Renamed: checkpoint-1167 -> checkpoint-9234 (eval_f1=92.34%)
```

### Checkpoint Directory:
```
checkpoints/phobert_finetuned/
├── checkpoint-8723/         # Epoch 1: F1 = 87.23%
├── checkpoint-9145/         # Epoch 2: F1 = 91.45%
├── checkpoint-9234/         # Epoch 3: F1 = 92.34% (BEST)
└── best_model/              # Copy of checkpoint-9234
```

## 🔧 Configuration

### Default (F1 Score với 4 chữ số):
```python
checkpoint_callback = SimpleMetricCheckpointCallback(
    metric_name='eval_f1',    # Sử dụng F1 score
    multiply_by=10000         # 4 chữ số (2 số thập phân)
)
```

### Alternative (Accuracy với 2 chữ số):
```python
checkpoint_callback = SimpleMetricCheckpointCallback(
    metric_name='eval_accuracy',  # Sử dụng accuracy
    multiply_by=100              # 2 chữ số (số nguyên)
)
```

## 📈 Ưu Điểm Format 4 Chữ Số

### 1. **Độ Phân Giải Cao**
```
Format 2 chữ số:
  checkpoint-87  (có thể là 87.23% hoặc 87.89%)
  checkpoint-91  (có thể là 91.12% hoặc 91.98%)
  → Không biết chính xác performance

Format 4 chữ số:
  checkpoint-8723  (chính xác 87.23%)
  checkpoint-9145  (chính xác 91.45%)
  → Biết chính xác performance!
```

### 2. **Dễ So Sánh Checkpoints**
```bash
# List checkpoints sorted
ls -1 checkpoints/phobert_finetuned/ | sort -n

checkpoint-8234
checkpoint-8567
checkpoint-8723  ← Dễ thấy sự tiến bộ
checkpoint-9012
checkpoint-9145
checkpoint-9234  ← Best checkpoint!
```

### 3. **Tracking Training Progress**
```
Training log:
  Epoch 1: checkpoint-8234 (82.34%)
  Epoch 2: checkpoint-8567 (85.67%)
  Epoch 3: checkpoint-8723 (87.23%)  ← Tăng 1.56%
  Epoch 4: checkpoint-9012 (90.12%)  ← Tăng 2.89%
  Epoch 5: checkpoint-9234 (92.34%)  ← Tăng 2.22%
```

### 4. **Model Selection**
```python
# Tìm best checkpoint dễ dàng
import os

checkpoints = os.listdir('checkpoints/phobert_finetuned')
checkpoints = [c for c in checkpoints if c.startswith('checkpoint-')]
checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

best_checkpoint = checkpoints[-1]
print(f"Best: {best_checkpoint}")  # checkpoint-9234

f1_score = int(best_checkpoint.split('-')[1]) / 10000
print(f"F1: {f1_score:.4f}")  # 0.9234 (92.34%)
```

## 🎨 Visualization Ideas

### Progress Chart:
```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
checkpoints = [8234, 8567, 8723, 9012, 9234]
f1_scores = [c / 10000 for c in checkpoints]

plt.plot(epochs, f1_scores, marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training Progress')
plt.ylim(0.8, 1.0)
for i, (e, f) in enumerate(zip(epochs, f1_scores)):
    plt.annotate(f'{f:.4f}', (e, f), textcoords="offset points", 
                xytext=(0,10), ha='center')
plt.show()
```

## 📝 Code Changes

### checkpoint_renamer.py:
```python
def __init__(self, metric_name='eval_f1', multiply_by=10000):
    # Default: F1 score với 4 chữ số
    # 0.8753 × 10000 = 8753
```

### train_phobert_trainer.py:
```python
checkpoint_callback = SimpleMetricCheckpointCallback(
    metric_name='eval_f1',     # Changed from 'eval_accuracy'
    multiply_by=10000          # Changed from 100
)
```

## 🔍 Verification

### Check syntax:
```bash
python -m py_compile checkpoint_renamer.py
python -m py_compile train_phobert_trainer.py
```

### Test example:
```python
# F1 = 0.8753
metric_value = 0.8753
multiply_by = 10000
checkpoint_name = int(metric_value * multiply_by)
print(f"checkpoint-{checkpoint_name}")  # checkpoint-8753

# Display
display_pct = metric_value * 100
print(f"{display_pct:.2f}%")  # 87.53%
```

## ✅ Ready to Use

Format checkpoint theo F1 score (4 chữ số) đã sẵn sàng:

```bash
python train_phobert_trainer.py
```

Output:
```
📁 Renamed: checkpoint-389 -> checkpoint-8753 (eval_f1=87.53%)
```

---

**Checkpoint format: F1 × 10000 = Name (ví dụ: 0.8753 → checkpoint-8753)**
