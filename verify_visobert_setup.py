"""
Verify setup for ViSoBERT training
"""
import os
import pandas as pd
import yaml


def check_config():
    """Check if config.yaml exists and is valid"""
    print("=" * 60)
    print("1. Configuration Check")
    print("=" * 60)
    
    if not os.path.exists("config.yaml"):
        print("[ERROR] config.yaml not found")
        return False
    
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("[OK] Config file loaded")
    print(f"  - Model: {config['model']['name']}")
    print(f"  - Num labels: {config['model']['num_labels']}")
    print(f"  - Max length: {config['model']['max_length']}")
    print(f"  - Train batch size: {config['training']['per_device_train_batch_size']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    print(f"  - Epochs: {config['training']['num_train_epochs']}")
    
    return config


def check_data_files(config):
    """Check if data files exist and have correct format"""
    print("\n" + "=" * 60)
    print("2. Data Files Check")
    print("=" * 60)
    
    files = {
        'train': config['paths']['train_file'],
        'validation': config['paths']['validation_file'],
        'test': config['paths']['test_file']
    }
    
    all_good = True
    
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"[ERROR] {name} file not found: {path}")
            all_good = False
            continue
        
        try:
            df = pd.read_csv(path)
            
            # Check columns
            required_cols = ['sentence', 'aspect', 'sentiment']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                print(f"[ERROR] {name}: Missing columns {missing}")
                all_good = False
                continue
            
            # Check sentiment values
            valid_sentiments = ['Positive', 'Negative', 'Neutral']
            invalid = df[~df['sentiment'].isin(valid_sentiments)]['sentiment'].unique()
            
            if len(invalid) > 0:
                print(f"[ERROR] {name}: Invalid sentiments {invalid}")
                all_good = False
                continue
            
            print(f"[OK] {name}: {len(df)} samples")
            print(f"     Sentiments: {df['sentiment'].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            all_good = False
    
    return all_good


def check_directories(config):
    """Check if output directories exist"""
    print("\n" + "=" * 60)
    print("3. Output Directories Check")
    print("=" * 60)
    
    dirs = [
        config['paths']['data_dir'],
        os.path.dirname(config['paths']['output_dir']),
        os.path.dirname(config['paths']['evaluation_report']),
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path}")
        else:
            print(f"[MISSING] {dir_path} (will be created during training)")


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("4. Dependencies Check")
    print("=" * 60)
    
    all_good = True
    
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[ERROR] PyTorch not installed")
        all_good = False
    
    try:
        import transformers
        print(f"[OK] Transformers: {transformers.__version__}")
    except ImportError:
        print("[ERROR] Transformers not installed")
        all_good = False
    
    try:
        import pandas
        print(f"[OK] Pandas: {pandas.__version__}")
    except ImportError:
        print("[ERROR] Pandas not installed")
        all_good = False
    
    try:
        import sklearn
        print(f"[OK] Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("[ERROR] Scikit-learn not installed")
        all_good = False
    
    return all_good


def main():
    print("=" * 60)
    print("ViSoBERT ABSA Setup Verification")
    print("=" * 60)
    print()
    
    config = check_config()
    if not config:
        return
    
    data_ok = check_data_files(config)
    check_directories(config)
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if data_ok and deps_ok:
        print("[SUCCESS] All checks passed! Ready to train with ViSoBERT.")
        print("\nNext step: Create training script for ViSoBERT")
        print("  - Use HuggingFace Trainer API")
        print("  - Model: 5CD-AI/Vietnamese-Sentiment-visobert")
        print("  - 3-class classification (Positive/Negative/Neutral)")
    else:
        print("[FAILED] Some checks failed. Please fix issues above.")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
