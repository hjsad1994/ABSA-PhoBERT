@echo off
REM ============================================================
REM Multi-Label ABSA Complete Workflow
REM ============================================================

echo.
echo ============================================================
echo Multi-Label ABSA Training Pipeline
echo ============================================================
echo.

REM Step 1: Preprocess Data
echo.
echo ============================================================
echo Step 1: Preprocessing Data
echo ============================================================
echo.
python preprocess_data.py
if errorlevel 1 (
    echo [ERROR] Preprocessing failed!
    pause
    exit /b 1
)

REM Step 2: Oversample (Optional)
echo.
echo ============================================================
echo Step 2: Oversampling Training Data (Optional)
echo ============================================================
echo.
python oversample_train.py
if errorlevel 1 (
    echo [WARNING] Oversampling failed or skipped
)

REM Step 3: Train Model
echo.
echo ============================================================
echo Step 3: Training Multi-Label Model
echo ============================================================
echo.
python train_phobert_multilabel.py
if errorlevel 1 (
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

REM Complete
echo.
echo ============================================================
echo [SUCCESS] Multi-Label ABSA Training Complete!
echo ============================================================
echo.
echo Output locations:
echo   - data/train.csv, val.csv, test.csv
echo   - data/train_oversampled.csv (if oversampling enabled)
echo   - checkpoints/phobert_multilabel/
echo   - results/evaluation_report.txt
echo   - training_logs/training_log_*.txt
echo.

pause
