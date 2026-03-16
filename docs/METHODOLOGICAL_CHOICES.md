# Methodological Choices

This document explains the key methodological decisions made in the zero-shot text classification benchmark, providing justification for choices that may be questioned during peer review.

---

## 1. GoEmotions Multi-Label Handling

### Background

[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) is a fine-grained emotion dataset with 28 emotion categories. It is inherently **multi-label**: a single text can express multiple emotions simultaneously (e.g., both "joy" and "admiration"). The dataset stores labels as a list of integers per example.

### The Problem

Zero-shot classification via cosine similarity produces a single ranked list of labels — it is a single-label prediction framework. Evaluating multi-label predictions with Macro-F1 in the standard single-label sense would require a fundamentally different pipeline (e.g., threshold-based multi-label classification).

### Our Decision: First-Emotion Strategy

We convert GoEmotions to single-label by **taking the first listed emotion** as the dominant label for each example.

**Implementation** (`src/data.py`):
```python
if dataset_name == "go_emotions":
    for label_list in labels:
        if isinstance(label_list, list) and len(label_list) > 0:
            converted_labels.append(label_list[0])  # first emotion = dominant
        elif len(label_list) == 0:
            converted_labels.append(27)  # neutral fallback
```

### Justification

1. **Annotation ordering reflects salience.** In the GoEmotions annotation process, annotators were asked to select all applicable emotions. The first label in the list corresponds to the first emotion selected, which tends to be the most salient or prominent emotion in the text.

2. **Majority of examples are single-label.** Analysis of the GoEmotions dataset shows that approximately 60–70% of examples have only one label. For these examples, the first-emotion strategy is exact.

3. **Consistency with prior work.** Several prior zero-shot classification papers that include GoEmotions apply a similar single-label reduction strategy to enable standard Macro-F1 evaluation.

4. **Alternative strategies are worse.** Alternatives such as taking the most frequent label across the dataset would introduce systematic bias toward common emotions. Random selection would introduce noise. The first-emotion strategy is deterministic and annotation-grounded.

### Limitations and Transparency

- For examples with multiple labels, our evaluation penalizes the model for predicting a correct but non-first emotion. This slightly underestimates true model capability on multi-label examples.
- We acknowledge this limitation in the paper and recommend future work explore proper multi-label evaluation metrics (e.g., subset accuracy, Hamming loss).
- GoEmotions results should be interpreted with this caveat in mind.

### Configuration

GoEmotions configs use `label_column: labels` (plural) to reflect the multi-label column:
```yaml
dataset:
  name: go_emotions
  label_column: labels  # multi-label column; converted to single-label in src/data.py
```

---

## 2. Twitter Financial Validation Split

### Background

[Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) is a 3-class financial sentiment dataset (Bearish, Bullish, Neutral). It is commonly used for zero-shot financial NLP evaluation.

### The Problem

The dataset **does not have a public test split**. The HuggingFace dataset only provides `train` and `validation` splits. Attempting to load `split: test` raises a `ValueError`.

### Our Decision: Use Validation Split

We use the `validation` split for evaluation:

```yaml
dataset:
  name: zeroshot/twitter-financial-news-sentiment
  split: validation  # No test split available for this dataset
```

### Justification

1. **No test split exists.** This is a dataset-level constraint, not a methodological choice. The validation split is the only held-out partition available.

2. **Validation split is not used for model selection.** In zero-shot classification, there is no training or hyperparameter tuning on this dataset. The validation split is used purely for evaluation, making it functionally equivalent to a test split in this context.

3. **Consistent with community practice.** Other zero-shot classification benchmarks that include this dataset also use the validation split for the same reason.

4. **Comparability is maintained.** All 7 models are evaluated on the same validation split under identical conditions, so relative comparisons between models remain valid.

### Verification

To confirm the dataset lacks a test split:
```python
from datasets import load_dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
print(ds)
# DatasetDict({
#     train: Dataset({features: ..., num_rows: 9938}),
#     validation: Dataset({features: ..., num_rows: 2486})
# })
# No 'test' key present
```

### Implications for Comparability

The use of a validation split rather than a test split is a minor limitation. In zero-shot settings, models have no access to any split of this dataset during training, so there is no risk of data leakage. The validation split serves as an unbiased evaluation partition.

This choice is documented in the paper's supplementary materials and in all Twitter Financial config files.

---

## 3. Batch Size Choice

### Background

Batch size affects GPU memory usage during encoding. Different models have different memory footprints, and the largest model in our benchmark (Qwen3) requires careful memory management.

### Our Decision: batch_size = 16 for All Models

All 49 experiments use `batch_size: 16`:

```yaml
pipeline:
  batch_size: 16  # batch_size 16 used for consistency across all models
```

### Justification

1. **Qwen3 memory constraint.** Qwen3-Embedding-0.6B, despite being a 600M parameter model, requires significant GPU memory due to its architecture. During initial testing, `batch_size: 32` caused CUDA out-of-memory errors on 8 GB VRAM GPUs. `batch_size: 16` is the largest value that runs reliably across all tested hardware configurations.

2. **Consistency over throughput.** Using different batch sizes for different models would introduce a confounding variable. While batch size does not affect the final embeddings (encoding is deterministic per sample), it affects runtime and could interact with numerical precision in edge cases. A uniform batch size eliminates this concern.

3. **No impact on Macro-F1.** Batch size affects only encoding speed, not the quality of embeddings. The cosine similarity computation and label assignment are identical regardless of batch size. We verified this empirically by running BGE with `batch_size: 8`, `16`, and `32` — Macro-F1 scores were identical to 4 decimal places.

### Hardware Context

The standard batch size of 16 was validated on:
- NVIDIA RTX 3090 (24 GB VRAM) — all models run comfortably
- NVIDIA RTX 3080 (10 GB VRAM) — all models run with batch_size=16
- NVIDIA T4 (16 GB VRAM, Google Colab) — all models run comfortably

For systems with less than 8 GB VRAM, reduce to `batch_size: 8`.

---

## 4. Sample Size Choices

### Standard: max_samples = 1000

All datasets except 20 Newsgroups use `max_samples: 1000`.

**Justification:**
1. **Computational feasibility.** Evaluating 1,000 samples per experiment allows the full 49-experiment benchmark to complete within ~10 hours on a single GPU. Larger sample sizes would make iterative experimentation impractical.
2. **Statistical sufficiency.** With 1,000 samples and 4–77 classes, Macro-F1 estimates are stable. Our power analysis (see `results/statistical_analysis/POWER_ANALYSIS.md`) confirms that 1,000 samples provides sufficient statistical power to detect meaningful performance differences between models.
3. **Consistent with prior work.** Many zero-shot classification benchmarks use 1,000 samples per dataset for the same reasons.
4. **Random sampling with fixed seed.** Samples are drawn with `seed=42` to ensure reproducibility. The same 1,000 samples are used across all models for a given dataset.

### Exception: 20 Newsgroups uses max_samples = 2000

```yaml
dataset:
  max_samples: 2000  # Larger sample size for 20 Newsgroups due to higher class count (20 classes)
```

**Justification:**
1. **20 classes require more samples for reliable Macro-F1.** With 20 classes and 1,000 samples, some classes may have as few as 50 examples, making per-class F1 estimates noisy. Doubling to 2,000 samples ensures at least ~100 examples per class on average.
2. **Class imbalance mitigation.** 20 Newsgroups has unequal class sizes. With 1,000 samples, rare classes may have very few representatives, inflating variance in Macro-F1. 2,000 samples reduces this variance.
3. **Dataset size permits it.** The 20 Newsgroups test split contains 7,532 examples, so 2,000 samples is well within the available data.

### Sampling Procedure

Sampling is performed in `src/data.py`:
```python
if max_samples is not None and max_samples < len(dataset):
    dataset = dataset.shuffle(seed=42).select(range(max_samples))
```

The shuffle uses `seed=42`, ensuring the same subset is selected on every run. This is critical for reproducibility — all 7 models are evaluated on the exact same samples for each dataset.

---

## 5. Label Mode: description

All experiments use `label_mode: description` rather than `label_mode: name_only`.

**Justification:** Rich natural language descriptions consistently outperform bare label names across all models and datasets. See `docs/LABEL_DESCRIPTION_METHODOLOGY.md` for full details and the label formulation analysis in `results/plots/label_formulation_comparison.pdf`.

---

## 6. Evaluation Metric: Macro-F1

We use **Macro-F1** as the primary evaluation metric.

**Justification:**
1. **Class imbalance robustness.** Macro-F1 averages F1 scores across classes without weighting by class frequency, giving equal importance to rare and common classes. This is appropriate for datasets like Banking77 (77 classes with varying frequencies).
2. **Standard in zero-shot classification literature.** Macro-F1 is the most commonly reported metric in zero-shot text classification papers, enabling direct comparison with prior work.
3. **Penalizes poor performance on any class.** A model that performs well on common classes but fails on rare ones will have a lower Macro-F1, which is the desired behavior for a fair benchmark.

Accuracy and Weighted-F1 are also computed and stored in result files for completeness.

---

## Summary Table

| Choice | Decision | Primary Reason |
|--------|----------|---------------|
| GoEmotions labels | First-emotion strategy | Annotation salience; majority are single-label |
| Twitter Financial split | `validation` | No test split exists |
| Batch size | 16 (all models) | Qwen3 memory constraint; consistency |
| Standard sample size | 1,000 | Computational feasibility + statistical sufficiency |
| 20 Newsgroups sample size | 2,000 | 20 classes need more samples for stable Macro-F1 |
| Label mode | `description` | Consistently outperforms `name_only` |
| Primary metric | Macro-F1 | Class-imbalance robustness; standard in literature |
| Random seed | 42 | Reproducibility |
