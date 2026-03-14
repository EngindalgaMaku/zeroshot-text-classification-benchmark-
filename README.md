# Zero-Shot Text Classification Benchmark

A comprehensive benchmark study comparing state-of-the-art sentence embedding models on zero-shot text classification across 6 diverse datasets.

## 📊 Key Results

We evaluated **5 embedding models** on **6 benchmark datasets** using zero-shot classification:

### Best Performing Models

| Rank | Model | Avg F1 | Best On | Worst On |
|------|-------|--------|---------|----------|
| 🥇 | **Qwen3-4B** | **68.2%** | DBPedia (83.7%) | Yahoo (49.0%) |
| 🥈 | MPNet | 64.0% | AG News (81.1%) | Yahoo (49.3%) |
| 🥉 | Jina-v3 | 62.7% | Banking77 (70.7%) | Twitter-Fin (43.5%) |
| 4 | E5-large | 62.0% | AG News (84.3%) | Yahoo (47.4%) |
| 5 | BGE-M3 | 59.8% | AG News (77.9%) | Twitter-Fin (40.8%) |

### Dataset Difficulty Rankings

1. **AG News** (80.3% avg) - Easiest ✅
2. **DBPedia** (78.9% avg)
3. **Banking77** (66.7% avg)
4. **20 Newsgroups** (55.1% avg)
5. **Twitter Financial** (51.0% avg)
6. **Yahoo Answers** (48.0% avg) - Hardest ❌

## 🎯 Project Overview

This repository implements a **zero-shot text classification** framework that:
- ✅ Requires **no training data** for new tasks
- ✅ Uses only **label names or descriptions** for classification
- ✅ Evaluates **state-of-the-art embedding models** (2024-2026)
- ✅ Provides **publication-ready visualizations** and reports

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd zeroshot_nlp__new

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run a single experiment
python main.py --config experiments/exp_agnews_mpnet.yaml

# Skip if results already exist (useful when adding new models)
python main.py --config experiments/exp_agnews_new_model.yaml --skip-existing

# Generate visualizations
python scripts/generate_beautiful_plots.py

# Generate publication report
python scripts/generate_heatmap_report.py
```

## 📁 Project Structure

```
zeroshot_nlp__new/
├── src/                    # Source code
│   ├── data.py            # Dataset loading
│   ├── encoders.py        # Embedding models
│   ├── labels.py          # Label descriptions
│   ├── pipeline.py        # Zero-shot classification
│   ├── metrics.py         # Evaluation metrics
│   └── runner.py          # Experiment runner
├── experiments/           # Experiment configurations
│   ├── exp_agnews_*.yaml
│   ├── exp_20newsgroups_*.yaml
│   └── exp_twitter_*.yaml
├── notebooks/             # Analysis notebooks
│   ├── MASTER_COLAB_EXPERIMENTS.ipynb
│   ├── MULTI_DATASET_EXPERIMENTS.ipynb
│   └── 03_tables_and_plots_UPDATED.ipynb
├── results/              # Experiment results
│   ├── raw/             # JSON metrics
│   ├── plots/           # Visualizations
│   └── tables/          # Summary tables
├── reports/             # Publication-ready reports
│   └── F1_HEATMAP_PUBLICATION.pdf
├── scripts/             # Visualization & analysis scripts
│   ├── generate_beautiful_plots.py
│   ├── generate_heatmap_report.py
│   ├── generate_dataset_report.py
│   └── generate_tables_and_plots.py
└── docs/                # Documentation
```

## 🔬 Methodology

### Zero-Shot Classification Pipeline

1. **Encode Labels**: Convert label names/descriptions to embeddings
2. **Encode Texts**: Convert input texts to embeddings
3. **Compute Similarity**: Calculate cosine similarity between text and label embeddings
4. **Predict**: Assign text to the most similar label

### Models Evaluated

| Model | Type | Dimensions | Parameters |
|-------|------|------------|------------|
| **Qwen3-4B** | Transformer | 3584 | 4B |
| **BGE-M3** | Bi-encoder | 1024 | 568M |
| **E5-large** | Bi-encoder | 1024 | 335M |
| **Jina-v3** | Bi-encoder | 1024 | 570M |
| **MPNet** | Bi-encoder | 768 | 110M |

### Datasets

| Dataset | Classes | Domain | Test Size | Used |
|---------|---------|--------|-----------|------|
| **AG News** | 4 | News | 7,600 | 1,000 |
| **20 Newsgroups** | 20 | Discussion Forums | 7,532 | 2,000 |
| **DBPedia-14** | 14 | Knowledge Base | 70,000 | 1,000 |
| **Banking77** | 77 | Banking Intents | 3,080 | 1,000 |
| **Twitter Financial** | 3 | Financial Sentiment | 3,000 | 1,000 |
| **Yahoo Answers** | 10 | Q&A | 60,000 | 1,000 |

## 📈 Results & Analysis

### Performance Heatmap

See `reports/F1_HEATMAP_PUBLICATION.pdf` for publication-quality visualizations.

### Key Findings

1. **Qwen3-4B dominates** across most datasets (4/6 best scores)
2. **Larger models ≠ always better**: MPNet (110M) outperforms E5-large (335M) on average
3. **Dataset difficulty correlates with class count**: 
   - Yahoo (10 classes) harder than AG News (4 classes)
   - Banking77 (77 classes) surprisingly manageable due to clear intent descriptions
4. **Label descriptions matter**: Well-crafted descriptions significantly boost performance

## 🛠️ Custom Experiments

### Create a New Experiment

```yaml
# experiments/my_experiment.yaml
dataset:
  name: "your_dataset"
  split: "test"
  text_column: "text"
  label_column: "label"
  max_samples: 1000

encoder:
  name: "sentence-transformers/all-mpnet-base-v2"
  
label_descriptions:
  - "Description for class 0"
  - "Description for class 1"
  # ...
```

```bash
python main.py --config experiments/my_experiment.yaml
```

## 📊 Generate Reports

```bash
# Generate all visualizations
python scripts/generate_beautiful_plots.py

# Generate publication heatmap
python scripts/generate_heatmap_report.py

# Generate dataset statistics
python scripts/generate_dataset_report.py
```

## 🔧 Advanced Usage

### Few-Shot Classification

```python
from src.few_shot import FewShotClassifier

classifier = FewShotClassifier(
    model_name="sentence-transformers/all-mpnet-base-v2",
    num_shots=3  # 3 examples per class
)
```

See `notebooks/` for detailed examples.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{zeroshot_benchmark_2026,
  title={Zero-Shot Text Classification: A Comprehensive Benchmark},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/your-username/zeroshot_nlp__new}
}
```

## 📖 Documentation

- **Getting Started**: See `docs/QUICKSTART.md`
- **Adding New Models**: See `docs/ADDING_NEW_MODELS.md` 🆕
- **Dataset Information**: See `docs/DATASET_SIZE_INFO.md`
- **Label Methodology**: See `docs/LABEL_DESCRIPTION_METHODOLOGY.md`

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for the Datasets and Transformers libraries
- Sentence-Transformers for the excellent embedding framework
- All model authors: Qwen Team, BAAI (BGE), Microsoft (E5), Jina AI, Microsoft (MPNet)

---

**Star ⭐ this repo if you find it helpful!**