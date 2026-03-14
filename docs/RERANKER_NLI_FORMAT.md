# Reranker NLI Format for Zero-Shot Classification

## Problem
Rerankers (cross-encoders) were giving poor accuracy (~42%) because we were using them incorrectly.

## Root Cause
We were treating rerankers like retrieval models:
```
Input: [text, label_description]
```

But for **zero-shot classification**, rerankers should use **NLI (Natural Language Inference) format**:
```
Premise: text
Hypothesis: "This text is about {label}"
```

## The NLI Approach

### What is NLI?
Natural Language Inference determines if a hypothesis is:
- **Entailment**: hypothesis is true given the premise
- **Contradiction**: hypothesis is false given the premise  
- **Neutral**: cannot determine

### Zero-Shot Classification as NLI
We reframe classification as an entailment problem:

1. **Premise**: The text to classify
2. **Hypothesis**: "This text is about {label}"
3. **Score**: Entailment probability = label probability

### Example
```python
# Text to classify
text = "Apple announces new iPhone with advanced camera features"

# Labels
labels = ["technology", "sports", "politics"]

# NLI Format
pairs = [
    (text, "This text is about technology."),
    (text, "This text is about sports."),
    (text, "This text is about politics."),
]

# Reranker scores entailment for each pair
# Highest entailment score = predicted label
```

## Implementation

### Before (Wrong)
```python
# Direct text-label pairs
scores = reranker.score(text, ["technology", "sports", "politics"])
```

### After (Correct)
```python
# NLI hypothesis format
hypotheses = [
    "This text is about technology.",
    "This text is about sports.",
    "This text is about politics.",
]
scores = reranker.score(text, hypotheses)
```

### Smart Formatting
Our implementation automatically handles both modes:

```python
def format_as_nli_hypothesis(label_text: str) -> str:
    # If already a full sentence (description mode), use as-is
    if label_text.startswith("This "):
        return label_text
    
    # Otherwise, wrap in NLI format
    return f"This text is about {label_text}."
```

This means:
- **name_only** mode: "technology" → "This text is about technology."
- **description** mode: "This text is about science..." → unchanged

## Why This Works

### Cross-Encoder Architecture
Cross-encoders process pairs jointly:
```
[CLS] premise [SEP] hypothesis [SEP]
```

When trained on NLI data (MNLI, SNLI), they learn to:
1. Understand semantic relationships
2. Detect entailment/contradiction
3. Score relevance between premise and hypothesis

### Zero-Shot Transfer
NLI-trained models can zero-shot classify because:
- They understand "This text is about X" patterns
- They can judge if text content matches label semantics
- No task-specific fine-tuning needed

## Expected Performance

### Before (Direct Pairs)
- Accuracy: ~42%
- Problem: Model not trained for direct text-label matching

### After (NLI Format)
- Expected accuracy: 75-85%
- Reason: Leverages NLI training objective

## References

### Key Papers
1. **Yin et al.**: "Benchmarking Zero-shot Text Classification" (2019)
   - Introduced NLI-based zero-shot classification
   - Showed BART-MNLI effectiveness

2. **"In Defense of Cross-Encoders for Zero-Shot Retrieval"** (2022)
   - Cross-encoders outperform bi-encoders in zero-shot scenarios
   - Early query-document interactions are crucial

### Models Using This Approach
- facebook/bart-large-mnli
- microsoft/deberta-v3-large-mnli
- cross-encoder/nli-deberta-v3-large

## Usage in Our Benchmark

The `predict_reranker` function now automatically applies NLI formatting:

```python
# In src/pipeline_reranker.py
predictions, confidences, scores = predict_reranker(
    texts=texts,
    label_texts=label_descriptions,
    label_ids=label_ids,
    reranker=reranker,
    use_nli_format=True,  # Default: enabled
)
```

This should significantly improve reranker performance across all datasets.
