# Towards Reliable Zero-Shot Text Classification with Modern Embedding and Reranking Models

A systematic study of modern bi-encoder and reranker models for zero-shot text classification, with focus on label semantics, hybrid pipelines, and robustness analysis.

## 🎯 Project Goals

- Evaluate state-of-the-art bi-encoder and cross-encoder models for zero-shot classification
- Systematic analysis of label description design impact
- Two-stage hybrid pipeline proposal
- Robustness analysis across English and Turkish scenarios
- Error type and confidence score analysis

## 📁 Project Structure

```
zero_shot_reliable_cls/
├── README.md
├── requirements.txt
├── main.py                    # Main entry point
├── experiments/               # Experiment configurations
│   ├── exp_agnews_baseline.yaml
│   ├── exp_agnews_hybrid.yaml
│   └── exp_turkish_news.yaml
├── src/                       # Core implementation
│   ├── __init__.py
│   ├── config.py             # Configuration loader
│   ├── data.py               # Dataset loading
│   ├── labels.py             # Label definitions
│   ├── encoders.py           # Bi-encoder models
│   ├── rerankers.py          # Cross-encoder models
│   ├── pipeline.py           # Classification pipelines
│   ├── metrics.py            # Evaluation metrics
│   ├── runner.py             # Experiment runner
│   └── utils.py              # Utilities
├── notebooks/                 # Colab notebooks
│   ├── 01_run_experiments.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_tables_and_plots.ipynb
├── results/                   # Experiment results
│   ├── raw/                  # Raw predictions and metrics
│   ├── tables/               # Generated tables
│   └── plots/                # Generated plots
└── data_cache/               # Cached datasets
```

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repository
git clone <your-repo>
cd zero_shot_reliable_cls

# Install dependencies
pip install -r requirements.txt

# Run a basic experiment
python main.py --config experiments/exp_agnews_baseline.yaml

# Run hybrid pipeline experiment
python main.py --config experiments/exp_agnews_hybrid.yaml
```

### Google Colab Setup

1. Open `notebooks/01_run_experiments.ipynb` in Google Colab
2. Mount your Google Drive
3. Upload the project folder to Drive
4. Follow the notebook instructions

## 📊 Experiment Configuration

Each experiment is defined by a YAML config file in `experiments/`:

```yaml
experiment_name: agnews_bge_description

dataset:
  name: ag_news
  split: test
  max_samples: 1000

task:
  label_mode: description  # name_only, description, multi_description

models:
  biencoder:
    name: BAAI/bge-m3
  reranker:
    name: jinaai/jina-reranker-v2-base-multilingual

pipeline:
  mode: hybrid  # biencoder, reranker, hybrid
  top_k: 3
```

## 🏷️ Label Modes

Three label representation modes are supported:

1. **name_only**: Simple class names (`["world", "sports", "business", "technology"]`)
2. **description**: Single descriptive sentence per class
3. **multi_description**: Multiple paraphrased descriptions per class

## 🔬 Models

### Bi-Encoders
- BAAI/bge-m3
- jinaai/jina-embeddings-v3
- sentence-transformers/all-mpnet-base-v2

### Rerankers
- jinaai/jina-reranker-v2-base-multilingual
- BAAI/bge-reranker-v2-m3
- cross-encoder/ms-marco-MiniLM-L-12-v2

## 📈 Results

Results are automatically saved to `results/raw/`:
- `{experiment_name}_metrics.json`: Accuracy, macro F1, per-class metrics
- `{experiment_name}_predictions.csv`: Detailed predictions with confidence scores

## 🔍 Analysis

Use the provided notebooks for:
- Error analysis (high-confidence mistakes)
- Confusion matrix visualization
- Label mode comparison
- Hybrid vs single-stage comparison

## 📝 Citation

```bibtex
@article{yourname2025reliable,
  title={Towards Reliable Zero-Shot Text Classification with Modern Embedding and Reranking Models},
  author={Your Name},
  journal={Target Journal},
  year={2025}
}
```

## 📄 License

MIT License