# Single-Label vs Multi-Label ABSA

So sánh hai approaches cho Aspect-Based Sentiment Analysis với PhoBERT.

## 📊 Quick Comparison

| Aspect | Single-Label | Multi-Label |
|--------|-------------|-------------|
| **Format** | Sentence-aspect pairs | Sentences with binary labels |
| **Samples** | 15,569 pairs (expanded) | 9,129 sentences (original) |
| **Labels** | 3 (Neg/Neu/Pos) | 33 (11 aspects × 3 sentiments) |
| **Loss** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Metrics** | Accuracy, Precision, Recall, F1 | F1 micro/macro, Hamming Loss, Exact Match |
| **Oversampling** | ✅ Yes (per-aspect) | ❌ No (not needed) |
| **Training Time** | ~25-30 min | ~20-25 min |
| **Expected F1** | ~0.92 | ~0.88 (micro) |

## 🎯 Detailed Comparison

### 1. Data Format

#### Single-Label

```csv
sentence,aspect,sentiment
"Pin tốt, camera đẹp",Battery,Positive
"Pin tốt, camera đẹp",Camera,Positive
"Màn hình ok",Display,Neutral
```

**Characteristics:**
- One row per sentence-aspect pair
- Expanded from original data
- Clear 1:1 mapping (sentence-aspect → sentiment)

#### Multi-Label

```csv
sentence,label_0,label_1,label_2,...,label_32
"Pin tốt, camera đẹp",0,0,1,0,0,1,0,0,0,...,0
"Màn hình ok",0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

**Characteristics:**
- One row per sentence
- 33 binary labels (11 aspects × 3 sentiments)
- Preserves original sentence structure

### 2. Training Process

#### Single-Label

```
dataset.csv (9,129 reviews)
         ↓
Expand to sentence-aspect pairs
         ↓
15,569 pairs
         ↓
Split train/val/test
         ↓
Oversample per aspect (balance labels within each aspect)
         ↓
Train/val/test: 21,060 / 1,556 / 1,558
         ↓
Train PhoBERT (3-class classification)
```

#### Multi-Label

```
dataset.csv (9,129 reviews)
         ↓
Convert to 33 binary labels
         ↓
9,129 samples
         ↓
Split train/val/test
         ↓
Train/val/test: 7,303 / 913 / 913
         ↓
Train PhoBERT (multi-label classification)
```

### 3. Model Architecture

#### Single-Label

```python
Input: "[Sentence] </s></s> [Aspect]"
         ↓
PhoBERT Encoder
         ↓
Classification Head (hidden_size → 3)
         ↓
Softmax
         ↓
Output: [P(Neg), P(Neu), P(Pos)]  # Sum to 1
```

#### Multi-Label

```python
Input: "[Sentence]"
         ↓
PhoBERT Encoder
         ↓
Classification Head (hidden_size → 33)
         ↓
Sigmoid (per label)
         ↓
Output: [P(label_0), P(label_1), ..., P(label_32)]  # Independent
```

### 4. Loss Functions

#### Single-Label

```python
CrossEntropyLoss
- Multi-class classification
- Mutually exclusive (one label per sample)
- Softmax normalization
```

#### Multi-Label

```python
BCEWithLogitsLoss
- Binary classification per label
- Labels are independent
- Sigmoid per label
```

### 5. Evaluation Metrics

#### Single-Label

```python
Metrics:
- Accuracy: Overall correctness
- Precision: True positives / Predicted positives
- Recall: True positives / Actual positives
- F1: Harmonic mean of precision/recall
- Per-class metrics (Negative, Neutral, Positive)

Example Output:
  Accuracy: 0.9234
  Precision: 0.9180
  Recall: 0.9150
  F1: 0.9165
```

#### Multi-Label

```python
Metrics:
- F1 (micro): Treat all labels equally
- F1 (macro): Average F1 across labels
- Hamming Loss: Fraction of incorrect labels
- Exact Match: All labels must be correct

Example Output:
  F1 (micro): 0.8800
  F1 (macro): 0.8500
  Hamming Loss: 0.0300
  Exact Match: 0.6500
```

### 6. Prediction Process

#### Single-Label

```python
# For one sentence with one aspect
input = "Pin tốt </s></s> Battery"
output = model(input)  # [0.05, 0.10, 0.85]
prediction = argmax(output)  # 2 (Positive)

# Need multiple calls for multiple aspects
results = []
for aspect in aspects:
    input = f"{sentence} </s></s> {aspect}"
    prediction = model(input)
    results.append((aspect, prediction))
```

#### Multi-Label

```python
# For one sentence (all aspects at once)
input = "Pin tốt, camera đẹp"
output = model(input)  # [0.02, 0.10, 0.88, 0.05, 0.12, 0.82, ...]
predictions = (output > 0.5)  # [0, 0, 1, 0, 0, 1, 0, ...]

# Decode active labels
active = []
for i, pred in enumerate(predictions):
    if pred == 1:
        aspect_idx = i // 3
        sentiment_idx = i % 3
        active.append((aspects[aspect_idx], sentiments[sentiment_idx]))
# Output: [('Battery', 'Positive'), ('Camera', 'Positive')]
```

### 7. Advantages & Disadvantages

#### Single-Label

**Advantages:**
- ✅ Fine-grained per-aspect analysis
- ✅ Easy to interpret (one prediction per aspect)
- ✅ Can balance labels per aspect (oversampling)
- ✅ Higher per-aspect accuracy
- ✅ Standard classification metrics

**Disadvantages:**
- ❌ Data expansion (more samples to process)
- ❌ Multiple forward passes for multiple aspects
- ❌ Loses sentence-level context
- ❌ Longer training time

#### Multi-Label

**Advantages:**
- ✅ Natural sentence structure preserved
- ✅ One forward pass per sentence (efficient)
- ✅ Joint learning across aspects
- ✅ Fewer samples (faster training)
- ✅ Better for overall sentiment

**Disadvantages:**
- ❌ Label imbalance (some labels very rare)
- ❌ Lower per-aspect accuracy
- ❌ Harder to interpret (33 probabilities)
- ❌ Threshold tuning required
- ❌ Multi-label metrics less intuitive

## 🎯 When to Use Each Approach

### Use Single-Label When:

1. **Need fine-grained analysis**
   - Want to analyze each aspect independently
   - Need high accuracy per aspect
   - Doing aspect-specific sentiment classification

2. **Have imbalanced aspects**
   - Some aspects have very few samples
   - Need per-aspect oversampling
   - Want balanced training per aspect

3. **Simple prediction interface**
   - One aspect at a time
   - Easy-to-interpret results
   - Standard classification pipeline

### Use Multi-Label When:

1. **Preserve natural structure**
   - Want to keep original sentences intact
   - Analyze multiple aspects together
   - Care about sentence-level sentiment

2. **Efficiency is important**
   - Have many samples
   - Need fast inference (one pass per sentence)
   - Limited computational resources

3. **Joint learning benefits**
   - Aspects are correlated
   - Want shared representations
   - Learn aspect interactions

## 📊 Performance Comparison

### Expected Results

| Metric | Single-Label | Multi-Label |
|--------|-------------|-------------|
| **F1 Score** | 0.92 | 0.88 (micro) |
| **Accuracy** | 0.93 | 0.65 (exact match) |
| **Training Time** | 25-30 min | 20-25 min |
| **Inference Time** | 11 ms × 11 aspects | 11 ms per sentence |
| **Memory Usage** | 6.5 GB VRAM | 6.2 GB VRAM |

### Label-Level Performance

**Single-Label:**
- Positive: F1 = 0.95
- Negative: F1 = 0.92
- Neutral: F1 = 0.88

**Multi-Label:**
- Common labels (e.g., General-Positive): F1 = 0.92
- Rare labels (e.g., Others-Neutral): F1 = 0.60
- Overall (macro): F1 = 0.85

## 🔄 Workflow Comparison

### Single-Label Workflow

```bash
cd single-label

# Step 1: Preprocess (expand to pairs)
python preprocess_data.py
# Output: 15,569 sentence-aspect pairs

# Step 2: Oversample (balance per aspect)
python oversample_train.py
# Output: 21,060 balanced pairs

# Step 3: Train
python train_phobert_trainer.py
# Output: Checkpoint-9234 (F1 = 92.34%)
```

### Multi-Label Workflow

```bash
cd multi-label

# Step 1: Preprocess (convert to binary labels)
python preprocess_data.py
# Output: 9,129 sentences with 33 labels

# Step 2: Train (no oversampling needed)
python train_phobert_multilabel.py
# Output: Best model (F1 micro = 88%)
```

## 💡 Recommendations

### For Research Papers

**Use Single-Label if:**
- Comparing with aspect-level baselines
- Reporting per-aspect metrics
- Standard ABSA benchmarks

**Use Multi-Label if:**
- Novel approach (less common in literature)
- Focusing on efficiency
- Multi-task learning angle

### For Production

**Use Single-Label if:**
- Need explainable predictions per aspect
- Aspect-specific decision making
- High accuracy is critical

**Use Multi-Label if:**
- Processing large volumes of text
- Need fast real-time inference
- Overall sentiment is enough

## 🧪 Experiment Both!

### Run Both Approaches

```bash
# Single-Label
cd single-label
run_all.bat

# Multi-Label
cd multi-label
run_all.bat

# Compare results
# Check: single-label/results/evaluation_report.txt
# Check: multi-label/results/evaluation_report.txt
```

### Key Questions to Answer

1. **Which has better F1?**
   - Single-label per-aspect F1
   - Multi-label micro/macro F1

2. **Which is faster?**
   - Training time
   - Inference time

3. **Which handles imbalance better?**
   - Check rare labels (Neutral, Packaging)
   - Check confusion matrices

4. **Which is more stable?**
   - Run with multiple seeds (42, 123, 456)
   - Compare standard deviation

## 📝 Summary

| Criteria | Winner | Reason |
|----------|--------|--------|
| **Accuracy** | Single-Label | Higher per-aspect F1 (0.92 vs 0.88) |
| **Speed** | Multi-Label | Faster training & inference |
| **Interpretability** | Single-Label | Clear predictions per aspect |
| **Efficiency** | Multi-Label | Fewer samples, one pass |
| **Naturalness** | Multi-Label | Preserves sentence structure |
| **Flexibility** | Single-Label | Easy to balance & tune per aspect |

**Best Overall:** Depends on use case!
- **For research/accuracy:** Single-Label
- **For production/speed:** Multi-Label

---

**Both approaches are ready to use! Try both and compare results.** 🚀
