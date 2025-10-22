"""
Script to verify that the setup is correct before training
"""
import os
import pandas as pd
from pathlib import Path
import yaml


def check_file_exists(file_path, description):
    """Check if a file exists"""
    exists = os.path.exists(file_path)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {file_path}")
    return exists


def check_data_format(file_path):
    """Check if data has correct format"""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['sentence', 'aspect', 'sentiment']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  [ERROR] Missing columns: {missing}")
            return False
        
        # Check sentiment values
        valid_sentiments = ['Positive', 'Negative', 'Neutral']
        invalid_sentiments = df[~df['sentiment'].isin(valid_sentiments)]['sentiment'].unique()
        
        if len(invalid_sentiments) > 0:
            print(f"  [ERROR] Invalid sentiments found: {invalid_sentiments}")
            return False
        
        print(f"  [OK] Format correct: {len(df)} rows")
        print(f"    - Sentiments: {df['sentiment'].value_counts().to_dict()}")
        print(f"    - Aspects: {len(df['aspect'].unique())} unique aspects")
        return True
    except Exception as e:
        print(f"  [ERROR] Error reading file: {e}")
        return False


def main():
    print("=" * 60)
    print("PhoBERT ABSA Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check config file
    print("\n1. Configuration")
    config_exists = check_file_exists("config.yaml", "Config file")
    all_good = all_good and config_exists
    
    if config_exists:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"  - Model: {config['model']['pretrained_model']}")
            print(f"  - Num labels: {config['model']['num_labels']}")
            print(f"  - Batch size: {config['training']['batch_size']}")
            print(f"  - Learning rate: {config['training']['learning_rate']}")
    
    # Check data files
    print("\n2. Data Files")
    train_exists = check_file_exists("data/train.csv", "Training data")
    val_exists = check_file_exists("data/val.csv", "Validation data")
    test_exists = check_file_exists("data/test.csv", "Test data")
    
    all_good = all_good and train_exists and val_exists and test_exists
    
    # Check data format
    if train_exists:
        print("\n3. Training Data Format")
        train_ok = check_data_format("data/train.csv")
        all_good = all_good and train_ok
    
    if val_exists:
        print("\n4. Validation Data Format")
        val_ok = check_data_format("data/val.csv")
        all_good = all_good and val_ok
    
    if test_exists:
        print("\n5. Test Data Format")
        test_ok = check_data_format("data/test.csv")
        all_good = all_good and test_ok
    
    # Check model files
    print("\n6. Model Files")
    check_file_exists("model.py", "Model definition")
    check_file_exists("train.py", "Training script")
    check_file_exists("prepare_data.py", "Data preparation")
    check_file_exists("utils.py", "Utilities")
    check_file_exists("focal_loss.py", "Focal loss")
    
    # Check dependencies
    print("\n7. Dependencies")
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
    except ImportError:
        print("  [MISSING] PyTorch not installed")
        all_good = False
    
    try:
        import transformers
        print(f"  [OK] Transformers: {transformers.__version__}")
    except ImportError:
        print("  [MISSING] Transformers not installed")
        all_good = False
    
    try:
        import pandas
        print(f"  [OK] Pandas: {pandas.__version__}")
    except ImportError:
        print("  [MISSING] Pandas not installed")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("[SUCCESS] All checks passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python train.py")
    else:
        print("[FAILED] Some checks failed. Please fix the issues above.")
        print("\nIf dependencies are missing, install them with:")
        print("  pip install -r requirements.txt")
    print("=" * 60)


if __name__ == '__main__':
    main()
