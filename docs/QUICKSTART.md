# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Option 1: Local Setup

```bash
# 1. Clone/Navigate to the project
cd zero_shot_reliable_cls

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run your first experiment
python main.py --config experiments/exp_agnews_baseline.yaml

# 4. Check results
ls results/raw/
```

### Option 2: Google Colab Setup

1. **Upload to Google Drive**
   - Upload the entire project folder to your Google Drive
   - Recommended path: `MyDrive/zero_shot_reliable_cls`

2. **Open Colab Notebook**
   - Go to https://colab.research.google.com
   - File → Open notebook → Upload
   - Upload `notebooks/01_run_experiments.ipynb`

3. **Run the notebook**
   - Follow the cells in order
   - Mount your Drive when prompted
   - Results will be saved automatically

## 📊 First Experiment: AG News Classification

The baseline experiment uses:
- Dataset: AG News (4 classes: World, Sports, Business, Tech)
- Model: BAAI/bge-m3 (bi-encoder)
- Label Mode: description (single sentence per class)
- Samples: 1000 (for quick testing)

Expected results:
- Accuracy: ~85-90%
- Macro F1: ~85-90%
- Runtime: ~2-3 minutes on GPU

## 🔄 Next Steps

### 1. Try Different Label Modes

```bash
# Minimal labels
python main.py --config experiments/exp_agnews_name_only.yaml

# Full experiment
python main.py --config experiments/exp_agnews_baseline.yaml
```

### 2. Try Hybrid Pipeline

```bash
python main.py --config experiments/exp_agnews_hybrid.yaml
```

This adds a reranker on top of bi-encoder, typically improving F1 by 2-5%.

### 3. Analyze Results

```python
# In Python or Jupyter
import json
import pandas as pd

# Load metrics
with open("results/raw/agnews_bge_description_metrics.json") as f:
    metrics = json.load(f)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Load predictions
df = pd.read_csv("results/raw/agnews_bge_description_predictions.csv")
errors = df[~df["correct"]].sort_values("confidence", ascending=False)
print(errors.head(10))
```

### 4. Run Error Analysis

Open `notebooks/02_error_analysis.ipynb` in Colab/Jupyter to:
- See confusion matrices
- Analyze high-confidence errors
- Compare per-class performance

### 5. Generate Paper Tables

Open `notebooks/03_tables_and_plots.ipynb` to create:
- LaTeX tables for papers
- Publication-quality plots
- Comparison charts

## ⚙️ Customization

### Add Your Own Dataset

1. **Prepare your data** (CSV format):
```csv
text,label
"Your text here...",0
"Another text...",1
```

2. **Add label definitions** in `src/labels.py`:
```python
LABEL_SETS["your_dataset"] = {
    "description": {
        0: ["Description for class 0"],
        1: ["Description for class 1"],
    }
}
```

3. **Create config** (e.g., `experiments/exp_your_data.yaml`):
```yaml
experiment_name: your_experiment

dataset:
  name: your_dataset
  # ... rest of config
```

### Try Different Models

Edit the config file:

```yaml
models:
  biencoder:
    name: jinaai/jina-embeddings-v3  # or any sentence-transformers model
  reranker:
    name: BAAI/bge-reranker-v2-m3    # or any cross-encoder model
```

Popular choices:
- **Bi-encoders**: BAAI/bge-m3, jinaai/jina-embeddings-v3, sentence-transformers/all-mpnet-base-v2
- **Rerankers**: jinaai/jina-reranker-v2-base-multilingual, BAAI/bge-reranker-v2-m3

## 🐛 Troubleshooting

### "Out of Memory" Error

Reduce batch size or sample size:
```yaml
dataset:
  max_samples: 500  # Instead of 1000

# Or modify src/encoders.py to use smaller batch_size
```

### "Model not found" Error

Make sure you have internet connection. Models are downloaded from HuggingFace on first use and cached locally.

### Slow on CPU

Use Google Colab with GPU:
- Runtime → Change runtime type → GPU

### Drive Mount Issues in Colab

If Drive won't mount:
1. Clear browser cache
2. Try incognito/private mode
3. Re-authenticate Google Drive

## 📧 Questions?

If you encounter issues:
1. Check the README.md for detailed documentation
2. Review the example configs in `experiments/`
3. Look at the notebook examples in `notebooks/`

## 🎯 Recommended Workflow

For a complete research project:

**Week 1: Setup & Baseline**
- Run baseline experiments on AG News
- Validate setup works correctly
- Document initial results

**Week 2: Ablation Studies**
- Test different label modes
- Compare bi-encoder vs hybrid
- Analyze errors

**Week 3: Advanced Experiments**
- Add more datasets
- Try different models
- Cross-lingual experiments (if applicable)

**Week 4: Analysis & Writing**
- Generate all tables and plots
- Write up results
- Prepare paper draft

Good luck with your research! 🚀