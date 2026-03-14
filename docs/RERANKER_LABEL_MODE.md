# Reranker Label Mode Configuration

## Problem
Reranker experiments were showing low accuracy (~42%) due to suboptimal label representation.

## Root Cause
Rerankers (cross-encoders) were using `label_mode: "description"` which provides verbose label descriptions like:

```
"This text is about international events, global politics, diplomacy, conflicts, or world affairs."
```

This is problematic because:
1. **Too verbose**: Rerankers work best with concise, query-like text
2. **Semantic redundancy**: Cross-encoders already perform deep semantic matching
3. **Noise**: Long descriptions can introduce irrelevant tokens that confuse the model

## Solution
Changed reranker configs to use `label_mode: "name_only"` which provides concise labels:

```
"world"
"sports"
"business"
"science and technology"
```

## Why This Works

### Bi-encoders vs Rerankers
- **Bi-encoders**: Encode text and labels separately, then compute similarity
  - Need rich descriptions for better semantic representation
  - Use `label_mode: "description"`
  
- **Rerankers (Cross-encoders)**: Process text-label pairs jointly with attention
  - Already perform deep semantic matching
  - Work better with concise, query-like labels
  - Use `label_mode: "name_only"`

### Cross-Encoder Architecture
Rerankers use the format: `[CLS] text [SEP] label [SEP]`

The model learns to:
1. Attend between text and label tokens
2. Capture semantic relationships
3. Output a relevance score

Short labels are better because:
- Less noise in attention mechanism
- Faster inference
- More focused semantic matching

## Expected Performance Improvement
- Before (description mode): ~42% accuracy
- After (name_only mode): 75-85% accuracy (expected)

This aligns with how rerankers are typically used in:
- Information retrieval (query + document)
- Semantic search (short query + passage)
- Question answering (question + answer)

## Implementation
Updated `notebooks/RERANKER_EXPERIMENTS.ipynb`:
```python
"task": {
    "type": "zero_shot_classification",
    "label_mode": "name_only",  # Changed from "description"
    "language": "en"
}
```

## Additional Improvements
Also added sigmoid normalization to reranker scores in `src/rerankers.py`:
```python
if normalize:
    scores = 1 / (1 + np.exp(-scores))  # Convert logits to probabilities
```

This ensures scores are in [0, 1] range for better confidence metrics.
