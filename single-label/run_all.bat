@echo off
REM ========================================================================
REM Complete ABSA Workflow - Single-Label Approach
REM ========================================================================

echo ========================================================================
echo ABSA with PhoBERT - Complete Workflow
echo ========================================================================
echo.

REM Step 1: Preprocess Data
echo [Step 1/3] Preprocessing data...
python preprocess_data.py
if %errorlevel% neq 0 (
    echo ERROR: Preprocessing failed!
    pause
    exit /b %errorlevel%
)
echo.

REM Step 2: Oversample Training Data
echo [Step 2/3] Oversampling training data...
python oversample_train.py
if %errorlevel% neq 0 (
    echo ERROR: Oversampling failed!
    pause
    exit /b %errorlevel%
)
echo.

REM Step 3: Train Model
echo [Step 3/3] Training PhoBERT model...
python train_phobert_trainer.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed!
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================================================
echo SUCCESS: Workflow completed!
echo ========================================================================
echo.
echo Results:
echo   - Checkpoints: ..\checkpoints\phobert_finetuned\
echo   - Report: ..\results\evaluation_report.txt
echo   - Log: ..\logs\training_log_*.txt
echo.
pause
