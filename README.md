# Zero-Shot Text Classification Benchmark

A comprehensive benchmark comparing 7 state-of-the-art sentence embedding models on zero-shot text classification across 7 diverse datasets, prepared for TACL (Transactions of the Association for Computational Linguistics) submission.

## Key Results

We evaluated **7 embedding models** on **7 benchmark datasets** using zero-shot classification with Macro-F1 as the primary metric.

### Models Evaluated

| Model | HuggingFace ID | Architecture | Parameters |
|-------|---------------|--------------|------------|
| BGE-M3 | `BAAI/bge-m3` | Bi-encoder | 568M |
| E5-large | `intfloat/e5-large-v2` | Bi-encoder | 335M |
| Instructor | `hkunlp/instructor-large` | Instruction-tuned | 335M |
| Jina v3 | `jinaai/jina-embeddings-v3` | Bi-encoder | 570M |
| MPNET | `sentence-transformers/all-mpnet-base-v2` | Bi-encoder | 110M |
| Nomic | `nomic-ai/nomic-embed-text-v1.5` | Bi-encoder | 137M |
| Qwen3 | `Qwen/Qwen3-Embedding-0.6B` | Transformer | 600M |

### Datasets

| Dataset | Classes | Domain | Split | Samples |
|---------|---------|--------|-------|---------|
| AG News | 4 | News topics | test | 1,000 |
| Banking77 | 77 | Banking intents | test | 1,000 |
| DBpedia-14 | 14 | Knowledge base | test | 1,000 |
| GoEmotions | 28 | Fine-grained emotions | test | 1,000 |
| 20 Newsgroups | 20 | Forum topics | test | 2,000 |
| Twitter Financial | 3 | Financial sentiment | validation | 1,000 |
| Yahoo Answers | 10 | Q&A topics | test | 1,000 |

## Project Overview

This repository implements a **zero-shot text classification** framework that:
- Requires no training data — classification is driven by label descriptions only
- Evaluates 7 state-of-the-art embedding models (2024–2026) under identical conditions
- Uses Macro-F1 as the primary metric for fair multi-class comparison
- Provides publication-ready visualizations, statistical tests, and analysis scripts

## Software Dependencies

All experiments were run with the following exact package versions:

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| sentence-transformers | 5.2.3 |
| transformers | 5.2.0 |
| torch | 2.10.0 |
| datasets | 4.5.0 |
| scikit-learn | 1.8.0 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |
| accelerate | 1.13.0 |
| einops | 0.8.2 |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB |
| GPU | NVIDIA (CUDA 11.8+) | NVIDIA A100/RTX 3090+ |

**Expected runtimes per experiment (1,000 samples, GPU):**
- Small models (MPNET, Nomic): ~2–3 minutes
- Medium models (BGE, E5, Instructor, Jina): ~3–5 minutes
- Large models (Qwen3): ~8–12 minutes
- Full benchmark (49 experiments): ~6–10 hours on a single GPU

CPU-only execution is possible but significantly slower (~10× slower per experiment).

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd zeroshot_nlp__new
```

### 2. Create a Virtual Environment

```bash
# Python 3.10 or higher required
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify GPU Setup (Optional)

```bash
python check_gpu.py
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3090
CUDA version: 12.1
```

### 5. Run a Quick Smoke Test

```bash
python main.py --config experiments/exp_ag_news_bge.yaml
```

Expected output includes:
```
Zero-Shot Text Classification Experiment
Random seed: 42 (for reproducibility)
...
Macro F1: 0.XXXX
```

Results are saved to `results/raw/`.

## Quick Start

### Run a Single Experiment

```bash
python main.py --config experiments/exp_ag_news_mpnet.yaml
```

### Skip Already-Completed Experiments

```bash
python main.py --config experiments/exp_ag_news_mpnet.yaml --skip-existing
```

### Run All 49 Benchmark Experiments

```bash
python scripts/fix_and_run_all_experiments.py
```

### Regenerate All Figures and Tables

```bash
python scripts/regenerate_all.py
```

See `docs/EXPERIMENT_EXECUTION_GUIDE.md` for detailed instructions.

## Project Structure

```
zeroshot_nlp__new/
├── src/                          # Core source code
│   ├── config.py                 # Configuration loading
│   ├── data.py                   # Dataset loading and preprocessing
│   ├── encoders.py               # Bi-encoder model wrappers
│   ├── labels.py                 # Label descriptions for all datasets
│   ├── metrics.py                # Evaluation metrics
│   ├── pipeline.py               # Zero-shot classification pipeline
│   └── runner.py                 # Experiment runner
├── experiments/                  # Experiment configuration files (YAML)
│   ├── exp_ag_news_*.yaml        # AG News experiments (7 models)
│   ├── exp_banking77_*.yaml      # Banking77 experiments (7 models)
│   ├── exp_dbpedia_14_*.yaml     # DBpedia-14 experiments (7 models)
│   ├── exp_go_emotions_*.yaml    # GoEmotions experiments (7 models)
│   ├── exp_20newsgroups_*.yaml   # 20 Newsgroups experiments (7 models)
│   ├── exp_zeroshot_twitter_*.yaml  # Twitter Financial experiments (7 models)
│   ├── exp_yahoo_answers_*.yaml  # Yahoo Answers experiments (7 models)
│   └── label_formulation/        # Label mode comparison configs
├── scripts/                      # Analysis and visualization scripts
│   ├── regenerate_all.py         # Master regeneration script
│   ├── statistical_analysis.py   # Friedman + Nemenyi tests
│   ├── generate_publication_heatmap.py
│   ├── generate_critical_difference_diagram.py
│   ├── generate_label_formulation_figure.py
│   ├── generate_task_type_analysis.py
│   ├── analyze_model_stability.py
│   ├── visualize_model_stability.py
│   ├── analyze_task_characteristics.py
│   ├── visualize_task_characteristics.py
│   ├── analyze_error_patterns.py
│   └── generate_confusion_matrices.py
├── results/                      # Experiment outputs
│   ├── raw/                      # JSON result files per experiment
│   ├── tables/                   # CSV summary tables
│   ├── plots/                    # Generated figures (PDF/EPS/PNG)
│   ├── statistical_analysis/     # Statistical test outputs
│   ├── stability_analysis/       # Model stability outputs
│   ├── task_characteristics/     # Task characteristic data
│   └── archive/                  # Archived old results
├── docs/                         # Documentation
│   ├── EXPERIMENT_EXECUTION_GUIDE.md
│   ├── METHODOLOGICAL_CHOICES.md
│   ├── LABEL_DESCRIPTION_METHODOLOGY.md
│   └── TROUBLESHOOTING.md
├── notebooks/                    # Jupyter notebooks
├── main.py                       # Main experiment entry point
└── requirements.txt              # Python dependencies
```

## Methodology

### Zero-Shot Classification Pipeline

1. **Encode label descriptions** — each class label is represented as a rich natural language description and encoded into a dense vector
2. **Encode input texts** — each test example is encoded into the same embedding space
3. **Compute cosine similarity** — similarity between each text and all label embeddings
4. **Predict** — assign the label with the highest cosine similarity

### Reproducibility

All experiments use:
- **Random seed**: 42 (applied to Python `random`, NumPy, PyTorch, and CUDA)
- **CUDA deterministic mode**: enabled when GPU is available
- **Batch size**: 16 for all models (chosen to prevent Qwen memory overflow)
- **Label mode**: `description` (rich natural language descriptions)
- **Split**: `test` for all datasets except Twitter Financial (`validation`)

### GoEmotions Multi-Label Handling

GoEmotions is a multi-label dataset. We convert it to single-label by taking the first listed emotion as the dominant label. See `docs/METHODOLOGICAL_CHOICES.md` for full justification.

### Twitter Financial Split

Twitter Financial News Sentiment does not have a public test split. We use the `validation` split for evaluation. See `docs/METHODOLOGICAL_CHOICES.md` for details.

## Generating Results

### Statistical Analysis

```bash
python scripts/statistical_analysis.py
```

Outputs Friedman test results, Nemenyi post-hoc comparisons, and critical difference diagram to `results/statistical_analysis/`.

### Publication Figures

```bash
python scripts/regenerate_all.py
```

Regenerates all figures and tables in the correct dependency order.

## Documentation

- **Experiment Execution**: `docs/EXPERIMENT_EXECUTION_GUIDE.md`
- **Methodological Choices**: `docs/METHODOLOGICAL_CHOICES.md`
- **Label Descriptions**: `docs/LABEL_DESCRIPTION_METHODOLOGY.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Adding New Models**: `docs/ADDING_NEW_MODELS.md`

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in the config file:
```yaml
pipeline:
  batch_size: 8  # default is 16
```

### Model Download Fails

Models are downloaded from HuggingFace on first use. Set a custom cache directory:
```bash
export HF_HOME=./model_cache
```

Or pre-download manually:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Dataset Download Fails

Datasets are downloaded via HuggingFace `datasets`. Set a custom cache:
```bash
export HF_DATASETS_CACHE=./data_cache
```

### Slow on CPU

Use `--skip-existing` to avoid re-running completed experiments:
```bash
python main.py --config experiments/exp_ag_news_bge.yaml --skip-existing
```

For full troubleshooting, see `docs/TROUBLESHOOTING.md`.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{zeroshot_benchmark_tacl2026,
  title={Zero-Shot Text Classification: A Comprehensive Benchmark of Sentence Embedding Models},
  author={...},
  journal={Transactions of the Association for Computational Linguistics},
  year={2026}
}
```

## License

MIT License — see LICENSE file for details.

## Acknowledgments

- HuggingFace for the `datasets` and `transformers` libraries
- Sentence-Transformers for the embedding framework
- Model authors: BAAI (BGE), Microsoft (E5, MPNET), HKUNLP (Instructor), Jina AI, Nomic AI, Alibaba (Qwen)
