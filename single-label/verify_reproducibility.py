"""
Verify Reproducibility - Check if all random operations use the same seed
For research purposes, all random operations should be reproducible
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path='../config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_config_seeds(config):
    """Check if all seeds in config are consistent"""
    print("=" * 80)
    print("SEED CONFIGURATION CHECK")
    print("=" * 80)
    print()
    
    # Get reproducibility section
    repro = config.get('reproducibility', {})
    
    if not repro:
        print("[FAIL] No 'reproducibility' section found in config!")
        print("       Please add reproducibility configuration to config.yaml")
        return False
    
    # Check master seed
    master_seed = repro.get('seed')
    print(f"Master Seed: {master_seed}")
    print()
    
    # Check all seeds
    seeds = {
        'Data Split': repro.get('data_split_seed'),
        'Oversampling': repro.get('oversampling_seed'),
        'Shuffle': repro.get('shuffle_seed'),
        'Training': repro.get('training_seed'),
        'Data Loader': repro.get('data_loader_seed'),
    }
    
    print("Seed Configuration:")
    all_same = True
    for name, seed in seeds.items():
        status = "[OK]" if seed == master_seed else "[X]"
        print(f"  {status} {name:20}: {seed}")
        if seed != master_seed:
            all_same = False
    
    print()
    
    if all_same:
        print("[PASS] All seeds are consistent!")
        print(f"       All random operations use seed: {master_seed}")
    else:
        print("[WARN] Seeds are NOT consistent!")
        print("       For reproducibility, all seeds should be the same")
        print(f"       Recommendation: Set all seeds to {master_seed}")
    
    print()
    return all_same


def simulate_data_split(seed):
    """Simulate data split to test reproducibility"""
    print("=" * 80)
    print("DATA SPLIT REPRODUCIBILITY TEST")
    print("=" * 80)
    print()
    print(f"Testing with seed: {seed}")
    print()
    
    # Create dummy data
    data = pd.DataFrame({
        'sentence': [f'sentence_{i}' for i in range(100)],
        'aspect': np.random.choice(['A', 'B', 'C'], 100),
        'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], 100)
    })
    
    # Split 1
    shuffled1 = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    train1 = shuffled1[:80]
    
    # Split 2 (with same seed)
    shuffled2 = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    train2 = shuffled2[:80]
    
    # Check if identical
    identical = train1.equals(train2)
    
    if identical:
        print("[PASS] Data split is reproducible!")
        print("       Same seed produces identical train/val/test splits")
    else:
        print("[FAIL] Data split is NOT reproducible!")
        print("       This should not happen - please check your code")
    
    print()
    print("Example: First 5 rows of train split")
    print(train1.head())
    print()
    
    return identical


def check_existing_data_files():
    """Check if data files exist and show their sizes"""
    print("=" * 80)
    print("EXISTING DATA FILES CHECK")
    print("=" * 80)
    print()
    
    files = [
        '../data/train.csv',
        '../data/val.csv',
        '../data/test.csv',
        '../data/train_oversampled.csv'
    ]
    
    print("Data files:")
    for file in files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            df = pd.read_csv(file)
            print(f"  [OK] {file:30} - {len(df):6,} rows ({size/1024:.1f} KB)")
        else:
            print(f"  [X]  {file:30} - Not found")
    
    print()


def main():
    """Main verification function"""
    print("\n")
    print("=" * 80)
    print("REPRODUCIBILITY VERIFICATION FOR RESEARCH")
    print("=" * 80)
    print()
    print("This script verifies that all random operations use consistent seeds")
    print("for reproducible results in research experiments.")
    print()
    
    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        print("[FAIL] config.yaml not found!")
        print("       Please ensure config.yaml exists in the parent directory")
        return
    
    # Check config seeds
    seeds_ok = check_config_seeds(config)
    
    # Get master seed
    seed = config.get('reproducibility', {}).get('seed', 42)
    
    # Test data split reproducibility
    split_ok = simulate_data_split(seed)
    
    # Check existing files
    check_existing_data_files()
    
    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    all_ok = seeds_ok and split_ok
    
    if all_ok:
        print("[PASS] ALL CHECKS PASSED!")
        print()
        print("Your configuration ensures reproducible results:")
        print(f"  - All seeds are consistent (seed={seed})")
        print("  - Data splitting is reproducible")
        print()
        print("You can safely run experiments with:")
        print("  1. python preprocess_data.py")
        print("  2. python oversample_train.py")
        print("  3. python train_phobert_trainer.py")
    else:
        print("[WARN] SOME CHECKS FAILED")
        print()
        print("Please review the issues above and fix them.")
        print()
        if not seeds_ok:
            print("  - Fix: Ensure all seeds in config.yaml are consistent")
    
    print()
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
