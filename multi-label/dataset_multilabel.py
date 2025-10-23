"""
Multi-Label ABSA Dataset
Supports both binary format and text sentiment labels
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MultiLabelABSADataset(Dataset):
    """
    Multi-Label ABSA Dataset
    
    Supports two formats:
    
    1. Text format (recommended):
       CSV with columns: sentence (or data), Battery, Camera, ..., Others
       Where values are: Negative, Neutral, Positive (or empty for no sentiment)
       
    2. Binary format:
       CSV with columns: sentence, label_0, label_1, ..., label_32
       Where labels are 0/1 (33 binary labels = 11 aspects × 3 sentiments)
    
    Label mapping (4 classes per aspect):
       - Negative: 0
       - Neutral:  1
       - Positive: 2
       - None (empty): 3
    
    Args:
        file_path: Path to CSV file
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length
        format_type: 'binary', 'text', or 'auto' (auto-detect)
    """
    
    def __init__(self, file_path, tokenizer, max_length=256, format_type='auto'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.df = pd.read_csv(file_path, encoding='utf-8')
        
        # Aspect names
        self.aspects = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others'
        ]
        
        # Sentiment mapping: text -> index (4 classes per aspect)
        self.sentiment_to_idx = {
            'negative': 0,
            'neutral': 1, 
            'positive': 2,
            'none': 3  # No sentiment for this aspect
        }
        
        # Reverse mapping: index -> text
        self.idx_to_sentiment = {v: k for k, v in self.sentiment_to_idx.items()}
        
        # Auto-detect format if needed
        if format_type == 'auto':
            format_type = self._detect_format()
            print(f"Auto-detected format: {format_type}")
        
        self.format_type = format_type
        
        # Parse data based on format
        if format_type == 'binary':
            self._parse_binary_format()
        elif format_type == 'text':
            self._parse_text_format()
        else:
            raise ValueError(f"Invalid format_type: {format_type}. Must be 'binary', 'text', or 'auto'")
        
        print(f"Loaded {len(self)} samples from {file_path}")
        print(f"Format: {format_type}")
        print(f"Aspects: {len(self.aspects)}")
    
    def _detect_format(self):
        """Auto-detect data format based on columns"""
        cols = set(self.df.columns)
        
        # Check for text format FIRST (most common: sentence/data + aspect columns)
        aspect_cols = set(self.aspects)
        has_text_col = ('sentence' in cols) or ('data' in cols)
        if aspect_cols.issubset(cols) and has_text_col:
            return 'text'
        
        # Check for binary format (label_0 to label_32)
        binary_cols = {f'label_{i}' for i in range(33)}
        if binary_cols.issubset(cols):
            return 'binary'
        
        # Could not detect - show error with columns
        print(f"Error: Could not detect format. Columns found: {sorted(cols)}")
        print(f"Expected one of:")
        print(f"  1. Text format: 'sentence' (or 'data') + aspect columns {self.aspects}")
        print(f"  2. Binary format: 'sentence' + label_0 to label_32")
        raise ValueError(f"Unknown data format. Please check CSV columns.")
    
    def _parse_binary_format(self):
        """
        Parse binary format: 33 binary labels (0/1)
        Convert to: [num_aspects] with sentiment indices
        
        Logic:
        - For each aspect (11 total):
          - Check 3 binary labels (negative, neutral, positive)
          - If all 0 -> default to neutral (idx=2)
          - If multiple 1s -> take first active (priority: negative > neutral > positive)
        """
        # Convert sentences to list, handling NaN
        self.sentences = self.df['sentence'].fillna('').astype(str).tolist()
        self.labels = []
        
        # Check if binary labels exist
        expected_cols = [f'label_{i}' for i in range(33)]
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing binary label columns: {missing_cols[:5]}... (total: {len(missing_cols)})")
        
        # Convert binary labels to sentiment indices per aspect
        for idx in range(len(self.df)):
            aspect_sentiments = []
            
            for aspect_idx in range(11):  # 11 aspects
                # Get 3 binary labels for this aspect
                neg_idx = aspect_idx * 3 + 0  # Negative
                neu_idx = aspect_idx * 3 + 1  # Neutral
                pos_idx = aspect_idx * 3 + 2  # Positive
                
                neg = int(self.df.iloc[idx][f'label_{neg_idx}'])
                neu = int(self.df.iloc[idx][f'label_{neu_idx}'])
                pos = int(self.df.iloc[idx][f'label_{pos_idx}'])
                
                # Determine sentiment (priority: negative > neutral > positive)
                if neg == 1:
                    sentiment_idx = 0  # negative
                elif neu == 1:
                    sentiment_idx = 1  # neutral
                elif pos == 1:
                    sentiment_idx = 2  # positive
                else:
                    # No sentiment
                    sentiment_idx = 3  # none
                
                aspect_sentiments.append(sentiment_idx)
            
            self.labels.append(aspect_sentiments)
        
        self.labels = np.array(self.labels, dtype=np.int64)  # [num_samples, 11]
    
    def _parse_text_format(self):
        """
        Parse text format: aspect columns with text values
        
        CSV columns: sentence (or data), Battery, Camera, ..., Others
        Values: 'Negative', 'Neutral', 'Positive', or empty/NaN
        
        Mapping:
            Negative -> 0
            Neutral  -> 1
            Positive -> 2
            Empty/NaN -> 3 (no sentiment)
        """
        # Get text column (accept both 'sentence' and 'data')
        text_col = 'sentence' if 'sentence' in self.df.columns else 'data'
        
        # Convert sentences to list, handling NaN
        self.sentences = self.df[text_col].fillna('').astype(str).tolist()
        self.labels = []
        
        # Check if aspect columns exist
        missing_aspects = [asp for asp in self.aspects if asp not in self.df.columns]
        if missing_aspects:
            raise ValueError(f"Missing aspect columns: {missing_aspects}")
        
        for idx in range(len(self.df)):
            aspect_sentiments = []
            
            for aspect in self.aspects:
                value = self.df.iloc[idx][aspect]
                
                # Handle empty/NaN -> 'none' (no sentiment)
                if pd.isna(value) or str(value).strip() == '':
                    sentiment_idx = 3  # none
                else:
                    # Clean and lowercase
                    sentiment_text = str(value).strip().lower()
                    
                    # Map to index
                    if sentiment_text in self.sentiment_to_idx:
                        sentiment_idx = self.sentiment_to_idx[sentiment_text]
                    else:
                        # Unknown sentiment -> treat as 'none'
                        print(f"Warning: Unknown sentiment '{sentiment_text}' at row {idx}, aspect {aspect}. Using 'none'.")
                        sentiment_idx = 3  # none
                
                aspect_sentiments.append(sentiment_idx)
            
            self.labels.append(aspect_sentiments)
        
        self.labels = np.array(self.labels, dtype=np.int64)  # [num_samples, 11]
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            dict with keys:
                - input_ids: [max_length]
                - attention_mask: [max_length]
                - labels: [11] with sentiment indices (0=positive, 1=negative, 2=neutral)
        """
        sentence = self.sentences[idx]
        labels = self.labels[idx]  # [11]
        
        # Ensure sentence is string (handle NaN/None)
        if not isinstance(sentence, str):
            sentence = str(sentence) if sentence is not None else ""
        
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [max_length]
            'labels': torch.tensor(labels, dtype=torch.long)  # [11]
        }
    
    def get_sentiment_distribution(self):
        """
        Get sentiment distribution across all aspects
        
        Returns:
            dict with counts for each sentiment
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {self.idx_to_sentiment[idx]: count for idx, count in zip(unique, counts)}
        
        print(f"\nSentiment distribution across all aspects:")
        total = sum(distribution.values())
        for sentiment in ['negative', 'neutral', 'positive', 'none']:
            count = distribution.get(sentiment, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {sentiment:10s}: {count:6,} ({pct:5.2f}%)")
        
        return distribution


def test_dataset():
    """Test dataset loading"""
    from transformers import AutoTokenizer
    import os
    
    print("=" * 80)
    print("Testing MultiLabelABSADataset")
    print("=" * 80)
    
    # Test with binary format
    print("\n1. Testing binary format...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Get script directory and construct absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, 'data', 'train.csv')
    
    try:
        dataset = MultiLabelABSADataset(
            train_file,
            tokenizer,
            max_length=256,
            format_type='auto'  # Auto-detect format
        )
        
        print(f"\n[OK] Dataset loaded: {len(dataset)} samples")
        
        # Test first sample
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Labels: {sample['labels']}")
        
        # Get distribution
        dataset.get_sentiment_distribution()
        
        # Test a few samples
        print(f"\nSample predictions for first 3 samples:")
        for i in range(min(3, len(dataset))):
            labels = dataset.labels[i]
            print(f"\n  Sample {i}:")
            # Safely print Vietnamese text
            try:
                sentence_preview = dataset.sentences[i][:50]
                print(f"    Sentence: {sentence_preview}...")
            except UnicodeEncodeError:
                print(f"    Sentence: [Vietnamese text - {len(dataset.sentences[i])} chars]")
            
            for aspect_idx, aspect in enumerate(dataset.aspects):
                sentiment_idx = labels[aspect_idx]
                sentiment = dataset.idx_to_sentiment[sentiment_idx]
                print(f"    {aspect:<15}: {sentiment}")
        
        print(f"\n[OK] Binary format test passed!")
    
    except Exception as e:
        print(f"\n[FAIL] Binary format test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_dataset()
