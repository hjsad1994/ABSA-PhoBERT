# Test Training với Logging và Checkpoint Renaming

## Tính năng đã thêm vào train_phobert_trainer.py

### ✅ 1. TeeLogger - Ghi log ra cả console và file
- **File log**: `logs/training_log_YYYYMMDD_HHMMSS.txt`
- Log tất cả output (print, logger) ra cả console và file
- Timestamp tự động cho mỗi lần training

### ✅ 2. Checkpoint Renaming
- **Trước**: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- **Sau**: `checkpoint-87`, `checkpoint-91`, `checkpoint-93`
- Đặt tên checkpoint theo accuracy (%) để dễ nhận biết
- Callback: `SimpleMetricCheckpointCallback`

### ✅ 3. Training Loss Logging
- Log training loss sau khi training xong
- Log các metrics:
  - Training loss
  - Training time
  - Samples/second
  - Steps/second

### ✅ 4. Enhanced Summary
- In tổng kết chi tiết cuối cùng
- Bao gồm:
  - Training loss
  - Test F1
  - Đường dẫn model, report, predictions
  - Đường dẫn training log

## Cách chạy

```bash
python train_phobert_trainer.py
```

## Output mẫu

```
📝 Training log sẽ được lưu tại: logs/training_log_20251022_193000.txt

================================================================================
PhoBERT ABSA Training with HuggingFace Trainer
================================================================================

...

================================================================================
Setting up Callbacks
================================================================================
✓ Checkpoint Renamer: Will rename checkpoints by accuracy (e.g., checkpoint-91)
✓ Early Stopping: patience=3, threshold=0.001

================================================================================
Creating Trainer
================================================================================
✓ Trainer created successfully

================================================================================
🎯 STARTING TRAINING
================================================================================

Epoch 1/5: 100%|████████| 500/500 [05:23<00:00, 1.55it/s]

📁 Renamed: checkpoint-500 -> checkpoint-87 (eval_accuracy=0.8723)

Epoch 2/5: 100%|████████| 500/500 [05:20<00:00, 1.56it/s]

📁 Renamed: checkpoint-1000 -> checkpoint-91 (eval_accuracy=0.9145)

...

================================================================================
✅ TRAINING COMPLETED
================================================================================
✓ Training loss: 0.2345
✓ Training time: 1234.56s
✓ Samples/second: 17.05
✓ Steps/second: 0.54

================================================================================
Evaluation on Test Set
================================================================================
✓ Test Accuracy: 0.9234
✓ Test Precision: 0.9123
✓ Test Recall: 0.9145
✓ Test F1: 0.9134

...

================================================================================
🎉 TRAINING COMPLETE!
================================================================================

✓ Summary:
   • Model fine-tuned successfully
   • Training loss: 0.2345
   • Test F1: 0.9134
   • Best model saved: checkpoints/phobert_finetuned/best_model
   • Evaluation report: results/evaluation_report.txt
   • Predictions: results/test_predictions.csv

📝 Training log saved: logs/training_log_20251022_193000.txt
```

## Cấu trúc thư mục sau training

```
E:\ABSA-PhoBERT\
├── logs/
│   └── training_log_20251022_193000.txt  # ← Log file chi tiết
├── checkpoints/
│   └── phobert_finetuned/
│       ├── checkpoint-87/                 # ← Renamed by accuracy
│       ├── checkpoint-91/                 # ← Renamed by accuracy
│       ├── checkpoint-93/                 # ← Renamed by accuracy (best)
│       └── best_model/                    # ← Best model copy
├── results/
│   ├── evaluation_report.txt
│   └── test_predictions.csv
└── train_phobert_trainer.py
```

## Kiểm tra log file

```bash
# Xem toàn bộ log
cat logs/training_log_20251022_193000.txt

# Xem chỉ phần training metrics
grep "Training loss" logs/training_log_20251022_193000.txt

# Xem các checkpoint được renamed
grep "Renamed:" logs/training_log_20251022_193000.txt
```

## So sánh với script mẫu

| Tính năng | Script mẫu (ViSoBERT) | Script PhoBERT | Status |
|-----------|----------------------|----------------|--------|
| TeeLogger | ✅ | ✅ | ✅ Done |
| Checkpoint Renamer | ✅ | ✅ | ✅ Done |
| Training Loss Log | ✅ | ✅ | ✅ Done |
| Early Stopping | ✅ | ✅ | ✅ Done |
| Best Model Save | ✅ | ✅ | ✅ Done |
| Enhanced Summary | ✅ | ✅ | ✅ Done |

## Files đã tạo/cập nhật

1. **checkpoint_renamer.py** - Callback để đổi tên checkpoint
   - `SimpleMetricCheckpointCallback` - Đơn giản, đổi tên theo metric
   - `BestMetricCheckpointCallback` - Nâng cao, chỉ giữ checkpoint tốt nhất

2. **train_phobert_trainer.py** - Script training chính
   - Thêm TeeLogger class
   - Thêm setup_logging() function
   - Import checkpoint_renamer
   - Cập nhật main() với logging và callbacks
   - Log training loss và metrics chi tiết

## Next Steps

Sẵn sàng để training:

```bash
python train_phobert_trainer.py
```

Training sẽ:
1. Tạo log file tự động
2. Đổi tên checkpoint theo accuracy
3. Log training loss và metrics
4. Lưu best model
5. Tạo report và predictions
