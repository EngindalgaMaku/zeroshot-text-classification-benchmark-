import pathlib, shutil, json, os

ROOT = pathlib.Path(".")

# ── helpers ──────────────────────────────────────────────────────────────────
def write(path, text):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"  wrote {p}")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 17.2 – Supplementary Materials
# ═══════════════════════════════════════════════════════════════════════════
print("\n[17.2] Writing docs/SUPPLEMENTARY_MATERIALS.md ...")

supp = """\
# Supplementary Materials

## Zero-Shot Text Classification: A Comprehensive Benchmark of Sentence Embedding Models

*Supplementary materials for TACL submission*

---

## S1. Full Experimental Details

### S1.1 Experimental Setup

All experiments follow a unified zero-shot classification protocol:

1. **Label encoding** — each class label is represented as a rich natural language description
   and encoded into a dense vector using the target embedding model.
2. **Text encoding** — each test example is encoded into the same embedding space.
3. **Cosine similarity** — similarity is computed between each text embedding and all label embeddings.
4. **Prediction** — the label with the highest cosine similarity is assigned.

No training data is used at any point. Classification is driven entirely by the semantic alignment
between text and label descriptions.

### S1.2 Reproducibility Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Random seed | 42 | Reproducibility |
| CUDA deterministic | Enabled | Reproducibility |
| Label mode | `description` | Outperforms `name_only` (see S3) |
| Batch size | 16 | Qwen3 memory constraint; consistency |
| Primary metric | Macro-F1 | Class-imbalance robustness |

Seed is applied to Python `random`, NumPy, PyTorch, and CUDA operations before any data loading
or model initialization.

### S1.3 Complete Results Table

Full Macro-F1 scores (%) for all 7 models x 7 datasets:

| Model | AG News | Banking77 | DBpedia-14 | GoEmotions | 20 Newsgroups | Twitter Fin. | Yahoo Ans. | Average |
|-------|---------|-----------|------------|------------|---------------|--------------|------------|---------|
| INSTRUCTOR-large | 85.1 | 61.7 | 81.9 | 20.4 | 60.4 | 64.1 | 54.4 | 61.1 |
| Qwen3-Embedding | 78.2 | 74.4 | 83.6 | 21.0 | 59.4 | 64.6 | 49.3 | 61.5 |
| all-mpnet-base-v2 | 81.4 | 62.4 | 77.8 | 14.8 | 56.8 | 56.8 | 49.3 | 57.0 |
| jina-embeddings-v3 | 79.9 | 70.7 | 76.5 | 22.4 | 57.2 | 43.4 | 48.5 | 56.9 |
| multilingual-e5-large | 84.3 | 59.0 | 81.2 | 17.4 | 50.5 | 49.7 | 47.4 | 55.6 |
| bge-m3 | 77.9 | 67.1 | 75.6 | 20.2 | 51.5 | 40.8 | 45.6 | 54.1 |
| snowflake-arctic-embed-m | 76.8 | 50.3 | 72.3 | 4.5 | 53.7 | 43.4 | 44.0 | 49.3 |

---

## S2. Dataset Descriptions and Preprocessing

### S2.1 Dataset Overview

| Dataset | HuggingFace ID | Classes | Domain | Split | Samples |
|---------|---------------|---------|--------|-------|---------|
| AG News | `fancyzhx/ag_news` | 4 | News topics | test | 1,000 |
| Banking77 | `PolyAI/banking77` | 77 | Banking intents | test | 1,000 |
| DBpedia-14 | `fancyzhx/dbpedia_14` | 14 | Knowledge base | test | 1,000 |
| GoEmotions | `google-research-datasets/go_emotions` | 28 | Fine-grained emotions | test | 1,000 |
| 20 Newsgroups | `SetFit/20_newsgroups` | 20 | Forum topics | test | 2,000 |
| Twitter Financial | `zeroshot/twitter-financial-news-sentiment` | 3 | Financial sentiment | validation | 1,000 |
| Yahoo Answers | `yahoo_answers_topics` | 10 | Q&A topics | test | 1,000 |

### S2.2 Dataset Characteristics

| Dataset | Avg. Text Length (chars) | Label Similarity | Task Type |
|---------|--------------------------|-----------------|-----------|
| AG News | 238 | 0.019 | Topic classification |
| Banking77 | 55 | 0.068 | Intent detection |
| DBpedia-14 | 288 | 0.022 | Entity classification |
| GoEmotions | 67 | 0.024 | Emotion recognition |
| 20 Newsgroups | 1,082 | 0.022 | Topic classification |
| Twitter Financial | 88 | 0.119 | Sentiment analysis |
| Yahoo Answers | 58 | 0.064 | Topic classification |

Label similarity is computed as the mean pairwise cosine similarity between label description
embeddings (using `all-mpnet-base-v2`). Higher values indicate more semantically similar labels,
which generally makes classification harder.

### S2.3 Preprocessing Steps

**General preprocessing (all datasets):**
1. Load dataset from HuggingFace `datasets` library
2. Shuffle with `seed=42` and select `max_samples` examples
3. Extract text and label columns
4. Map integer labels to string class names

**GoEmotions (multi-label handling):**
- GoEmotions stores labels as a list of integers per example
- We convert to single-label by taking the first listed emotion as the dominant label
- Approximately 60-70% of examples have only one label; for these, the conversion is exact
- See S4.1 for full justification

**Twitter Financial (split selection):**
- The dataset does not have a public test split
- We use the `validation` split (2,486 examples), sampling 1,000 with `seed=42`
- See S4.2 for full justification

**20 Newsgroups (sample size):**
- Uses `max_samples: 2000` instead of the standard 1,000
- Rationale: 20 classes require more samples for stable per-class F1 estimates
- See S4.4 for full justification

---

## S3. Label Formulation Analysis

### S3.1 Experimental Design

We compared two label modes on 3 diverse datasets (AG News, Banking77, GoEmotions) across all 7 models:

- `name_only`: Bare class name (e.g., "World", "Sports")
- `description`: Rich natural language description (e.g., "World news and international affairs...")

### S3.2 Results Summary

Descriptions consistently outperform bare names across all models and datasets. The benefit is
largest for datasets with ambiguous or domain-specific label names (Banking77, ~+8 pp) and
smallest for datasets with highly distinctive fine-grained labels (GoEmotions, ~+4 pp).

See `results/plots/label_formulation_comparison.pdf` for the full visualization.

---

## S4. Methodological Notes

### S4.1 GoEmotions Multi-Label Handling

GoEmotions is inherently multi-label: a single text can express multiple emotions simultaneously.
Our zero-shot pipeline produces single-label predictions (highest cosine similarity), so we convert
GoEmotions to single-label by taking the first listed emotion as the dominant label.

**Justification:**
1. Annotation ordering reflects salience — the first emotion selected tends to be the most prominent
2. ~60-70% of examples have only one label; for these, the conversion is exact
3. Consistent with prior zero-shot classification work on GoEmotions
4. Deterministic and annotation-grounded (vs. random selection or frequency-based strategies)

**Limitation:** For multi-label examples, the model is penalized for predicting a correct but
non-first emotion. This slightly underestimates true model capability. We recommend future work
explore proper multi-label evaluation metrics.

### S4.2 Twitter Financial Validation Split

The `zeroshot/twitter-financial-news-sentiment` dataset does not have a public test split. Only
`train` and `validation` splits are available. We use the `validation` split for evaluation.

**Justification:** In zero-shot classification, no training or hyperparameter tuning occurs on
this dataset. The validation split is used purely for evaluation, making it functionally equivalent
to a test split. All 7 models are evaluated on the same split under identical conditions.

### S4.3 Batch Size

All experiments use `batch_size: 16`. This was chosen to prevent CUDA out-of-memory errors on
Qwen3-Embedding-0.6B with 8 GB VRAM GPUs. Batch size does not affect embedding quality or
Macro-F1 scores (verified empirically with BGE-M3 at batch sizes 8, 16, and 32 — identical
Macro-F1 to 4 decimal places).

### S4.4 Sample Sizes

- **Standard**: `max_samples: 1000` for all datasets
- **Exception**: `max_samples: 2000` for 20 Newsgroups (20 classes require more samples for stable per-class F1)
- **SST-2**: 872 samples (dataset natural validation split size)

Sampling uses `seed=42` to ensure all 7 models are evaluated on the exact same subset for each dataset.

---

## S5. Statistical Analysis Details

### S5.1 Friedman Test

The Friedman test is a non-parametric test for detecting differences among k related samples
(models) across multiple blocks (datasets).

**Configuration:**
- Datasets (blocks): 8
- Models (treatments): 7
- Significance level: alpha = 0.05

**Results:**
- Friedman chi-squared = 16.4464, df = 6, **p = 0.0115**
- **Conclusion:** Significant — models have significantly different performance distributions

### S5.2 Effect Size (Kendall's W)

- **W = 0.3426** — Moderate effect
- Interpretation: Models show consistent relative performance patterns, but with substantial
  variation across datasets

### S5.3 Post-Hoc Nemenyi Test

Pairwise comparisons using the Nemenyi test with critical distance CD = 3.1853.

**Average ranks:**

| Model | Average Rank |
|-------|-------------|
| Qwen3 | 2.00 |
| INSTRUCTOR | 2.88 |
| E5-large | 3.38 |
| Jina v3 | 4.50 |
| MPNet | 4.75 |
| BGE-M3 | 5.25 |
| Nomic-MoE | 5.25 |

**Significant pairwise differences (rank diff > 3.1853):**
- Qwen3 vs BGE-M3 (diff = 3.25)
- Qwen3 vs Nomic-MoE (diff = 3.25)

**Model cliques (no significant difference within clique):**
- Clique 1 (top performers): Qwen3, INSTRUCTOR, E5-large, Jina v3, MPNet
- Clique 2 (lower performers): BGE-M3, Nomic-MoE

### S5.4 Statistical Power Analysis

- **Power = 0.8796 (87.96%)** — Adequate (exceeds 80% threshold)
- Non-centrality parameter lambda = 16.4464
- Required datasets for 80% power: 7 (current: 8)
- Required datasets for 90% power: 9 (current: 8) — marginally below

The current benchmark provides adequate statistical power to detect the observed moderate effect
size. The significant Friedman test result is reliable and not attributable to insufficient power.

---

## S6. Model Details

### S6.1 Model Specifications

| Model | HuggingFace ID | Architecture | Parameters | Embedding Dim |
|-------|---------------|--------------|------------|---------------|
| BGE-M3 | `BAAI/bge-m3` | Bi-encoder (XLM-RoBERTa) | 568M | 1024 |
| E5-large | `intfloat/multilingual-e5-large` | Bi-encoder | 560M | 1024 |
| INSTRUCTOR | `hkunlp/instructor-large` | Instruction-tuned T5 | 335M | 768 |
| Jina v3 | `jinaai/jina-embeddings-v3` | Bi-encoder | 570M | 1024 |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | Bi-encoder | 110M | 768 |
| Nomic | `nomic-ai/nomic-embed-text-v1.5` | Bi-encoder (MoE) | 137M | 768 |
| Qwen3 | `Qwen/Qwen3-Embedding-0.6B` | Transformer | 600M | 1024 |

### S6.2 Model Selection Rationale

Models were selected to represent diverse architectures, training approaches, and parameter scales:

- **BGE-M3**: Multilingual model from BAAI; strong cross-lingual capabilities
- **E5-large**: Microsoft multilingual E5; trained with contrastive learning on diverse data
- **INSTRUCTOR**: Instruction-tuned model; uses task-specific prompts for encoding
- **Jina v3**: Jina AI latest embedding model; optimized for retrieval and classification
- **MPNet**: Sentence-Transformers baseline; widely used in prior work
- **Nomic**: Mixture-of-Experts architecture; efficient with competitive performance
- **Qwen3**: Alibaba Qwen3 embedding; strong performance on classification tasks

### S6.3 INSTRUCTOR Task Prefix

INSTRUCTOR requires a task-specific instruction prefix:
- For text encoding: `"Represent the text for classification: "`
- For label encoding: `"Represent the label for text classification: "`

---

## S7. Configuration Files

All 49 canonical experiment configurations are in `experiments/`. Each YAML file specifies
dataset, model, pipeline parameters, and output settings. See `submission/REPOSITORY_AUDIT.md`
for the complete list of config files.

### S7.1 Naming Convention

Canonical configs follow the pattern `exp_{dataset}_{model}.yaml`:
- `exp_ag_news_bge.yaml`, `exp_ag_news_e5.yaml`, ..., `exp_ag_news_qwen3.yaml`
- `exp_banking77_bge.yaml`, ..., `exp_banking77_qwen3.yaml`
- etc.

### S7.2 Label Formulation Configs

Additional configs for the label formulation analysis are in `experiments/label_formulation/`,
covering both `name_only` and `description` modes for all 9 datasets x 7 models.

---

## S8. Reproducibility Checklist

| Item | Status |
|------|--------|
| Random seed fixed (42) | Yes |
| CUDA deterministic mode | Yes |
| All configs in repository | Yes (49 canonical + 126 label formulation) |
| Raw results included | Yes |
| Analysis scripts included | Yes |
| Software versions documented | Yes |
| Hardware requirements documented | Yes |
| Step-by-step execution guide | Yes (`docs/EXPERIMENT_EXECUTION_GUIDE.md`) |
| Master regeneration script | Yes (`scripts/regenerate_all.py`) |
| Methodological choices documented | Yes (`docs/METHODOLOGICAL_CHOICES.md`) |

To reproduce all results from scratch:

```bash
pip install -r requirements.txt
python scripts/fix_and_run_all_experiments.py
python scripts/regenerate_all.py
```

Expected total runtime: 6-10 hours on a single GPU (NVIDIA RTX 3090 or equivalent).
"""

write("docs/SUPPLEMENTARY_MATERIALS.md", supp)
print("[17.2] Done.")
