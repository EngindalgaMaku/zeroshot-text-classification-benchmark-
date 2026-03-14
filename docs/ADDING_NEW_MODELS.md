# Adding New Embedding Models

This guide explains how to add new embedding models to your benchmark without re-running existing experiments.

## 🎯 Problem

You've already run experiments with 5 models across 6 datasets. Now you want to add 2-3 more models, but you **don't want to re-run** the existing experiments (which would waste time and resources).

## ✅ Solution: `--skip-existing` Flag

The framework supports skipping experiments that already have results.

---

## 📝 Step-by-Step Guide

### 1. Add Your New Model Configuration

Create a new experiment config for your new model:

```yaml
# experiments/exp_agnews_new_model.yaml
experiment_name: "ag_news_new_model"

dataset:
  name: "fancyzhx/ag_news"
  split: "test"
  text_column: "text"
  label_column: "label"
  max_samples: 1000

models:
  biencoder:
    name: "your-new-model-name"  # e.g., "BAAI/bge-large-en-v1.5"

task:
  label_mode: "description"

pipeline:
  normalize_embeddings: true

output:
  output_dir: "results"
  save_metrics: true
  save_predictions: true
```

### 2. Run With `--skip-existing` Flag

```bash
# This will SKIP experiments that already have results
python main.py --config experiments/exp_agnews_new_model.yaml --skip-existing
```

**What happens:**
- ✅ Checks if `results/raw/ag_news_new_model_metrics.json` exists
- ✅ If exists: Skips the experiment (saves time!)
- ✅ If doesn't exist: Runs the experiment

### 3. Run Multiple Experiments (Batch)

If you want to run your new model across all 6 datasets:

**Linux/Mac:**
```bash
#!/bin/bash
# run_new_model.sh

for dataset in agnews banking77 dbpedia 20newsgroups yahoo twitter_financial
do
    python main.py --config experiments/exp_${dataset}_new_model.yaml --skip-existing
done
```

**Windows (PowerShell):**
```powershell
# run_new_model.ps1

$datasets = @("agnews", "banking77", "dbpedia", "20newsgroups", "yahoo", "twitter_financial")

foreach ($dataset in $datasets) {
    python main.py --config "experiments/exp_${dataset}_new_model.yaml" --skip-existing
}
```

---

## 🔄 Example Workflow

### Scenario: You want to add "BAAI/bge-large-en-v1.5"

1. **Create configs for all datasets:**
   ```
   experiments/exp_agnews_bge_large.yaml
   experiments/exp_banking77_bge_large.yaml
   experiments/exp_dbpedia_bge_large.yaml
   experiments/exp_20newsgroups_bge_large.yaml
   experiments/exp_yahoo_bge_large.yaml
   experiments/exp_twitter_financial_bge_large.yaml
   ```

2. **Run with skip flag:**
   ```bash
   # Will only run the new model, skips existing results
   python scripts/run_all_experiments.py --skip-existing
   ```

3. **Generate updated visualizations:**
   ```bash
   python scripts/generate_beautiful_plots.py
   python scripts/generate_heatmap_report.py
   ```

---

## 📊 Use Cases

### Use Case 1: Adding a Single New Model
```bash
# Just run the new model configs
python main.py --config experiments/exp_agnews_new_model.yaml --skip-existing
```

### Use Case 2: Re-running Failed Experiments
```bash
# If some experiments failed, delete their results and re-run
rm results/raw/failed_experiment_metrics.json
python main.py --config experiments/failed_experiment.yaml --skip-existing
```

### Use Case 3: Updating Existing Results
```bash
# Without --skip-existing, it will overwrite
python main.py --config experiments/exp_agnews_mpnet.yaml
```

---

## 🎨 Visualization Updates

After adding new models:

```bash
# Regenerate all plots with new model included
python scripts/generate_beautiful_plots.py

# Update publication heatmap
python scripts/generate_heatmap_report.py

# Update dataset statistics
python scripts/generate_dataset_report.py
```

The scripts automatically detect all models in `results/raw/` and include them in visualizations.

---

## 💡 Tips

1. **Naming Convention:** Keep experiment names consistent:
   ```
   <dataset>_<model>_<variant>
   ```

2. **Check Results Directory:**
   ```bash
   ls results/raw/*_metrics.json
   ```
   
3. **Dry Run:** Use `--skip-existing` on your existing experiments first to verify it works:
   ```bash
   python main.py --config experiments/exp_agnews_mpnet.yaml --skip-existing
   # Should output: "⏭️  SKIPPING: ag_news_mpnet"
   ```

---

## 🔍 How It Works

The `--skip-existing` flag checks for the existence of:
```
results/raw/<experiment_name>_metrics.json
```

If this file exists, the experiment is skipped. Otherwise, it runs normally.

**Implementation in `src/runner.py`:**
```python
def run_experiment(cfg: Dict[str, Any], skip_existing: bool = False):
    if skip_existing:
        metrics_file = Path(output_dir) / "raw" / f"{exp_name}_metrics.json"
        if metrics_file.exists():
            print(f"⏭️  SKIPPING: {exp_name}")
            return None
    # ... rest of experiment
```

---

## 🚀 Quick Reference

| Command | Effect |
|---------|--------|
| `python main.py --config <file>.yaml` | Run experiment (overwrite if exists) |
| `python main.py --config <file>.yaml --skip-existing` | Skip if results exist |
| `rm results/raw/<exp>_metrics.json` | Force re-run specific experiment |
| `python scripts/generate_*.py` | Update visualizations with new results |

---

**This feature ensures you never waste time re-running experiments that are already complete!** 🎉