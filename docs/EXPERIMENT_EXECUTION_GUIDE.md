# Experiment Execution Guide

This guide covers how to run individual experiments, batch experiments, and regenerate all analyses for the zero-shot text classification benchmark.

## Prerequisites

Ensure you have completed the setup described in `README.md`:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Verify installation
python -c "import sentence_transformers, torch, datasets; print('OK')"
```

---

## Running Individual Experiments

### Basic Usage

```bash
python main.py --config <path-to-config>
```

**Example — BGE on AG News:**
```bash
python main.py --config experiments/exp_ag_news_bge.yaml
```

**Expected output:**
```
======================================================================
Zero-Shot Text Classification Experiment
======================================================================
Random seed: 42 (for reproducibility)
Loading dataset: ag_news, split: test
Sampled 1000 examples
Loading bi-encoder: BAAI/bge-m3
Model loaded on device: cuda
Encoding 1000 texts...
Encoding 4 label descriptions...
Computing similarities...
----------------------------------------------------------------------
Results:
  Accuracy:  0.8760
  Macro F1:  0.8743
  Weighted F1: 0.8761
----------------------------------------------------------------------
Results saved to: results/raw/exp_ag_news_bge_metrics.json
Predictions saved to: results/raw/exp_ag_news_bge_predictions.csv
```

### Skip Already-Completed Experiments

Use `--skip-existing` to avoid re-running experiments that already have results:

```bash
python main.py --config experiments/exp_ag_news_bge.yaml --skip-existing
```

This is useful when adding new models to an existing benchmark run.

### Config File Structure

Each experiment is defined by a YAML config file:

```yaml
experiment_name: exp_ag_news_bge

dataset:
  name: ag_news           # HuggingFace dataset name
  split: test             # Dataset split (test/validation)
  text_column: text       # Column containing input text
  label_column: label     # Column containing labels
  max_samples: 1000       # Number of samples to evaluate

task:
  type: zero_shot_classification
  label_mode: description  # Use rich label descriptions
  language: en

models:
  biencoder:
    provider: hf
    name: BAAI/bge-m3     # HuggingFace model ID
  reranker: null

pipeline:
  mode: biencoder
  normalize_embeddings: true
  batch_size: 16           # Standard batch size for all models

evaluation:
  metrics:
    - accuracy
    - macro_f1
    - per_class_f1

output:
  save_predictions: true
  save_metrics: true
  output_dir: results/raw
```

### All Available Experiment Configs

The benchmark includes 49 core experiment configs (7 models × 7 datasets):

| Dataset | Config Pattern |
|---------|---------------|
| AG News | `experiments/exp_ag_news_{model}.yaml` |
| Banking77 | `experiments/exp_banking77_{model}.yaml` |
| DBpedia-14 | `experiments/exp_dbpedia_14_{model}.yaml` |
| GoEmotions | `experiments/exp_go_emotions_{model}.yaml` |
| 20 Newsgroups | `experiments/exp_20newsgroups_{model}.yaml` |
| Twitter Financial | `experiments/exp_zeroshot_twitter_financial_news_sentiment_{model}.yaml` |
| Yahoo Answers | `experiments/exp_yahoo_answers_topics_{model}.yaml` |

Where `{model}` is one of: `bge`, `e5`, `instructor`, `jina_v3`, `mpnet`, `nomic`, `qwen3`.

---

## Running Batch Experiments

### Run All 49 Core Experiments

```bash
python scripts/fix_and_run_all_experiments.py
```

This script:
- Runs all 49 model-dataset combinations sequentially
- Skips experiments that already have results
- Logs errors and continues on failure
- Reports progress and estimated time remaining

**Expected runtime:** 6–10 hours on a single GPU (varies by model size).

### Run Experiments for a Specific Model

```bash
# Run all BGE experiments
for config in experiments/exp_*_bge.yaml; do
    python main.py --config "$config" --skip-existing
done
```

On Windows (PowerShell):
```powershell
Get-ChildItem experiments\exp_*_bge.yaml | ForEach-Object {
    python main.py --config $_.FullName --skip-existing
}
```

### Run Experiments for a Specific Dataset

```bash
# Run all AG News experiments
for config in experiments/exp_ag_news_*.yaml; do
    python main.py --config "$config" --skip-existing
done
```

### Run Label Formulation Experiments

Label formulation experiments compare `name_only` vs `description` label modes:

```bash
# Run label formulation comparison experiments
python scripts/run_label_formulation_experiments.py
```

Configs are in `experiments/label_formulation/`.

---

## Checking Experiment Results

### List Completed Experiments

```bash
ls results/raw/*.json
```

Each completed experiment produces:
- `results/raw/{experiment_name}_metrics.json` — Macro-F1, accuracy, per-class F1
- `results/raw/{experiment_name}_predictions.csv` — per-sample predictions

### Inspect a Result File

```python
import json

with open("results/raw/exp_ag_news_bge_metrics.json") as f:
    metrics = json.load(f)

print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Collect All Results into a Summary Table

```bash
python scripts/collect_all_results.py
```

Outputs `results/MULTI_DATASET_RESULTS.csv` with all experiments in a single table.

---

## Regenerating All Analyses

### Master Regeneration Script

To regenerate all figures, tables, and analyses from raw results:

```bash
python scripts/regenerate_all.py
```

This runs all analysis scripts in the correct dependency order with progress logging.

**Expected output:**
```
[00:00] Starting full analysis regeneration
[00:01] Step 1/8: Collecting results... done (2.3s)
[00:03] Step 2/8: Statistical analysis... done (18.4s)
[00:21] Step 3/8: Publication heatmap... done (4.1s)
[00:25] Step 4/8: Critical difference diagram... done (3.2s)
...
[05:42] All analyses complete. Total time: 5m 42s
```

### Running Individual Analysis Scripts

You can also run each analysis script independently:

#### Statistical Analysis (Friedman + Nemenyi)
```bash
python scripts/statistical_analysis.py
```
Outputs to `results/statistical_analysis/`:
- `friedman_test_results.txt`
- `nemenyi_pairwise_pvalues.csv`
- `critical_difference_diagram.pdf`

#### Publication Heatmap
```bash
python scripts/generate_publication_heatmap.py
```
Outputs to `results/plots/`:
- `publication_heatmap.pdf`
- `publication_heatmap.eps`

#### Critical Difference Diagram
```bash
python scripts/generate_critical_difference_diagram.py
```
Outputs to `results/plots/`:
- `critical_difference_diagram.pdf`
- `critical_difference_diagram.eps`

#### Label Formulation Analysis
```bash
python scripts/generate_label_formulation_figure.py
```
Outputs to `results/plots/`:
- `label_formulation_comparison.pdf`

#### Task Type Analysis
```bash
python scripts/generate_task_type_analysis.py
```
Outputs to `results/plots/`:
- `task_type_analysis.pdf`

#### Model Stability Analysis
```bash
# Step 1: Compute stability metrics
python scripts/analyze_model_stability.py

# Step 2: Generate stability visualization
python scripts/visualize_model_stability.py
```
Outputs to `results/stability_analysis/`.

#### Task Characteristics Analysis
```bash
# Step 1: Compute task characteristics
python scripts/analyze_task_characteristics.py

# Step 2: Generate scatter plots
python scripts/visualize_task_characteristics.py
```
Outputs to `results/task_characteristics/`.

#### Error Pattern Analysis
```bash
# Step 1: Analyze error patterns
python scripts/analyze_error_patterns.py

# Step 2: Generate confusion matrices
python scripts/generate_confusion_matrices.py
```
Outputs to `results/plots/`.

---

## Expected Outputs

After running all experiments and analyses, the following files should exist:

### Raw Results (`results/raw/`)
- 49 `*_metrics.json` files (one per model-dataset combination)
- 49 `*_predictions.csv` files

### Summary Tables (`results/tables/`)
- `main_results_table.csv` — Macro-F1 scores for all 49 experiments
- `label_formulation_comparison.csv` — name_only vs description comparison
- `model_stability_ranking.csv` — models ranked by coefficient of variation

### Figures (`results/plots/`)
- `publication_heatmap.pdf` / `.eps` — model × dataset Macro-F1 heatmap
- `critical_difference_diagram.pdf` / `.eps` — statistical ranking diagram
- `label_formulation_comparison.pdf` — label mode impact visualization
- `task_type_analysis.pdf` — performance by task type
- `stability_performance_scatter.pdf` — mean performance vs stability
- `confusion_matrix_*.pdf` — confusion matrices for key datasets

### Statistical Analysis (`results/statistical_analysis/`)
- `friedman_test_results.txt`
- `nemenyi_pairwise_pvalues.csv`
- `POWER_ANALYSIS.md`

---

## Experiment Configuration Reference

### Standard Parameters (All Experiments)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `seed` | 42 | Set in `main.py` before config loading |
| `batch_size` | 16 | Chosen for Qwen memory compatibility |
| `max_samples` | 1000 | Standard; 20 Newsgroups uses 2000 |
| `label_mode` | `description` | Rich natural language descriptions |
| `split` | `test` | Except Twitter Financial (`validation`) |
| `normalize_embeddings` | `true` | Required for cosine similarity |

### Dataset-Specific Overrides

| Dataset | Parameter | Value | Reason |
|---------|-----------|-------|--------|
| 20 Newsgroups | `max_samples` | 2000 | 20 classes require more samples for reliable Macro-F1 |
| Twitter Financial | `split` | `validation` | No public test split available |
| GoEmotions | `label_column` | `labels` | Multi-label column (converted to single-label) |

---

## Troubleshooting Experiments

### Experiment Fails with OOM Error

Reduce batch size in the config:
```yaml
pipeline:
  batch_size: 8
```

### Experiment Produces NaN Metrics

Check that the dataset loaded correctly:
```python
import json
with open("results/raw/your_experiment_metrics.json") as f:
    print(json.load(f))
```

If `macro_f1` is `null` or `NaN`, the experiment likely failed silently. Re-run without `--skip-existing`.

### Results Differ Between Runs

Ensure seed is being applied. Check `main.py` calls `set_seed(42)` before config loading. If using CUDA, verify `torch.backends.cudnn.deterministic = True` is set.

### Model Not Found

Models are downloaded from HuggingFace on first use. Ensure internet access or pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```
