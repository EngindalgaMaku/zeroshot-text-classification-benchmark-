# Repository Structure Audit

Generated for TACL submission package preparation.

## Core Source Code (`src/`)

All source code is properly organized:

| File | Purpose |
|------|---------|
| `src/config.py` | Configuration loading and validation |
| `src/data.py` | Dataset loading and preprocessing |
| `src/encoders.py` | Bi-encoder model wrappers |
| `src/labels.py` | Label descriptions for all datasets |
| `src/metrics.py` | Evaluation metrics (Macro-F1, accuracy) |
| `src/pipeline.py` | Zero-shot classification pipeline |
| `src/runner.py` | Experiment runner |
| `src/utils.py` | Utility functions |

## Experiment Configurations (`experiments/`)

### Core Benchmark Configs (49 files: 7 datasets × 7 models)

| Dataset | BGE | E5 | Instructor | Jina | MPNET | Nomic | Qwen3 |
|---------|-----|----|-----------|----|-------|-------|-------|
| AG News | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Banking77 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DBpedia-14 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GoEmotions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 Newsgroups | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Twitter Financial | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Yahoo Answers | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Total: 49/49 core configs present ✓**

### Extended Benchmark Configs (14 additional files)

- IMDB: 7 configs (exp_imdb_{model}.yaml)
- SST-2: 7 configs (exp_sst2_{model}.yaml)

### Label Formulation Configs (`experiments/label_formulation/`)

- 126 configs covering all 9 datasets × 7 models × 2 modes (name_only, description)

### Legacy/Non-Standard Configs

The following files in `experiments/` root use non-standard naming (no `exp_` prefix).
They are older configs and are superseded by the canonical `exp_*` versions:

- `ag_news_instructor.yaml`, `ag_news_jina_task.yaml`
- `banking77_instructor.yaml`, `banking77_jina_task.yaml`
- `dbpedia_14_instructor.yaml`, `dbpedia_14_jina_task.yaml`
- `SetFit_20_newsgroups_instructor.yaml`, `SetFit_20_newsgroups_jina_task.yaml`
- `yahoo_answers_topics_instructor.yaml`, `yahoo_answers_topics_jina_task.yaml`
- `zeroshot_twitter_financial_news_sentiment_instructor.yaml`
- `zeroshot_twitter_financial_news_sentiment_jina_task.yaml`

These are retained for reference but are not part of the canonical benchmark.

## Analysis Scripts (`scripts/`)

All analysis scripts are properly organized in `scripts/`. Key scripts:

| Script | Purpose |
|--------|---------|
| `regenerate_all.py` | Master script to regenerate all outputs |
| `statistical_analysis.py` | Friedman + Nemenyi tests |
| `generate_publication_heatmap.py` | Model × dataset heatmap |
| `generate_critical_difference_diagram.py` | CD diagram |
| `generate_label_formulation_figure.py` | Label mode comparison |
| `generate_task_type_analysis.py` | Task type analysis |
| `analyze_model_stability.py` | Stability metrics |
| `visualize_model_stability.py` | Stability visualizations |
| `analyze_task_characteristics.py` | Task characteristics |
| `visualize_task_characteristics.py` | Task characteristic plots |
| `analyze_error_patterns.py` | Error pattern analysis |
| `generate_confusion_matrices.py` | Confusion matrix figures |
| `collect_all_results.py` | Aggregate results to CSV |

## Root-Level Files

| File | Status | Notes |
|------|--------|-------|
| `main.py` | ✓ Keep | Main experiment entry point |
| `requirements.txt` | ✓ Keep | Python dependencies |
| `README.md` | ✓ Keep | Project documentation |
| `fix_runner.py` | ✓ Keep | Batch experiment runner |
| `hf_login.py` | ✓ Keep | HuggingFace authentication helper |
| `check_gpu.py` | ✓ Keep | GPU verification utility |
| `_p.py` | ⚠ Legacy | Scratch/utility file, not part of submission |
| `test_bge.py` | ⚠ Legacy | Ad-hoc test script, not part of submission |
| `test_gpu.py` | ⚠ Legacy | Ad-hoc test script, not part of submission |
| `run_nomic_experiments.bat` | ⚠ Legacy | Windows batch script, not part of submission |

## Summary

- **Core source code**: Properly organized in `src/` ✓
- **Analysis scripts**: Properly organized in `scripts/` ✓
- **Experiment configs**: All 49 core configs present in `experiments/` ✓
- **Results**: Organized in `results/` with subdirectories ✓
- **Documentation**: Organized in `docs/` ✓
- **Minor cleanup needed**: A few legacy root-level files (`_p.py`, `test_bge.py`, `test_gpu.py`) are not part of the submission package but are harmless to retain in the repository
