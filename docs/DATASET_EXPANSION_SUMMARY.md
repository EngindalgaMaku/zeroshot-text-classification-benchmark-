# Dataset Expansion Implementation Summary

## Overview

This document summarizes the dataset expansion evaluation and implementation for the TACL Benchmark Strengthening project. Two new datasets (IMDB and SST-2) have been added to address critical gaps in sentiment diversity and binary classification coverage.

## Decision Process

### Analysis Conducted

1. **Current Dataset Coverage Analysis** (`scripts/analyze_dataset_coverage.py`)
   - Evaluated 7 existing datasets across task types, domains, class counts, and text lengths
   - Identified gaps: sentiment diversity (WEAK), binary classification (LIMITED), long text (LIMITED)

2. **Candidate Dataset Evaluation** (`scripts/evaluate_candidate_datasets.py`)
   - Evaluated 3 candidates: IMDB, SST-2, TREC
   - Scored each on 6 criteria: task diversity, domain diversity, class diversity, text length, quality, adoption
   - IMDB scored 9.4/10, SST-2 scored 8.3/10, TREC scored 4.3/10

3. **Decision Framework** (`docs/DATASET_INCLUSION_FRAMEWORK.md`)
   - Created systematic framework with weighted scoring
   - Documented inclusion/exclusion criteria
   - Applied framework to all candidates

### Final Decision

**Selected:** IMDB + SST-2 (comprehensive sentiment coverage)

**Rationale:**
- IMDB (9.4/10): Fills TWO critical gaps (sentiment diversity + long text)
- SST-2 (8.3/10): Adds GLUE benchmark prestige, complements IMDB
- TREC (4.3/10): Rejected due to overlap with Yahoo Answers

## Implementation Details

### 1. Label Definitions Added

**File:** `src/labels.py`

Added label definitions for both datasets with `name_only` and `description` modes:

**IMDB:**
- 2 classes: negative, positive
- Description mode provides movie-specific sentiment descriptions

**SST-2:**
- 2 classes: negative, positive  
- Description mode provides general sentiment descriptions

### 2. Data Loading Updates

**File:** `src/data.py`

Added explicit handling for both datasets:
- IMDB: Direct HuggingFace dataset loading
- SST-2: GLUE benchmark loading (`glue/sst2`)

### 3. Experiment Configurations Created

**Total:** 14 new config files (7 models × 2 datasets)

**IMDB Configs:**
- `experiments/exp_imdb_mpnet.yaml`
- `experiments/exp_imdb_qwen3.yaml`
- `experiments/exp_imdb_snowflake.yaml`
- `experiments/exp_imdb_instructor.yaml`
- `experiments/exp_imdb_jina_v3.yaml`
- `experiments/exp_imdb_bge.yaml`
- `experiments/exp_imdb_e5.yaml`

**SST-2 Configs:**
- `experiments/exp_sst2_mpnet.yaml`
- `experiments/exp_sst2_qwen3.yaml`
- `experiments/exp_sst2_snowflake.yaml`
- `experiments/exp_sst2_instructor.yaml`
- `experiments/exp_sst2_jina_v3.yaml`
- `experiments/exp_sst2_bge.yaml`
- `experiments/exp_sst2_e5.yaml`

**Standardized Configuration:**
- batch_size: 16 (consistent across all models)
- max_samples: 1000 (standard sample size)
- split: test (IMDB), validation (SST-2)
- label_mode: description
- All configs follow established naming convention


## Benchmark Impact

### Before Expansion (7 datasets, 49 experiments)

**Task Type Coverage:**
- Topic classification: 3 datasets (STRONG)
- Sentiment classification: 1 dataset (WEAK)
- Entity classification: 1 dataset
- Intent classification: 1 dataset
- Emotion classification: 1 dataset

**Class Count Distribution:**
- Binary/Ternary (≤3): 1 dataset
- Few-class (4-5): 1 dataset
- Medium (6-15): 3 datasets
- Many-class (16-30): 1 dataset
- Fine-grained (30+): 1 dataset

**Text Length Distribution:**
- Short (<50 tokens): 4 datasets
- Medium (50-150): 2 datasets
- Long (150+): 1 dataset

### After Expansion (9 datasets, 63 experiments)

**Task Type Coverage:**
- Topic classification: 3 datasets (STRONG)
- Sentiment classification: 3 datasets (STRONG) ✅
- Entity classification: 1 dataset
- Intent classification: 1 dataset
- Emotion classification: 1 dataset

**Class Count Distribution:**
- Binary/Ternary (≤3): 3 datasets ✅
- Few-class (4-5): 1 dataset
- Medium (6-15): 3 datasets
- Many-class (16-30): 1 dataset
- Fine-grained (30+): 1 dataset

**Text Length Distribution:**
- Short (<50 tokens): 5 datasets
- Medium (50-150): 2 datasets
- Long (150+): 2 datasets ✅

**Key Improvements:**
- ✅ Sentiment coverage: WEAK → STRONG (1 → 3 datasets)
- ✅ Binary classification: LIMITED → ADEQUATE (1 → 3 datasets)
- ✅ Long text representation: LIMITED → ADEQUATE (1 → 2 datasets)
- ✅ Domain diversity: +1 domain (movie reviews)

## Running the Experiments

### Manual Execution

To run all new experiments manually:

```bash
# IMDB experiments (7 models)
python main.py --config experiments/exp_imdb_mpnet.yaml
python main.py --config experiments/exp_imdb_qwen3.yaml
python main.py --config experiments/exp_imdb_snowflake.yaml
python main.py --config experiments/exp_imdb_instructor.yaml
python main.py --config experiments/exp_imdb_jina_v3.yaml
python main.py --config experiments/exp_imdb_bge.yaml
python main.py --config experiments/exp_imdb_e5.yaml

# SST-2 experiments (7 models)
python main.py --config experiments/exp_sst2_mpnet.yaml
python main.py --config experiments/exp_sst2_qwen3.yaml
python main.py --config experiments/exp_sst2_snowflake.yaml
python main.py --config experiments/exp_sst2_instructor.yaml
python main.py --config experiments/exp_sst2_jina_v3.yaml
python main.py --config experiments/exp_sst2_bge.yaml
python main.py --config experiments/exp_sst2_e5.yaml
```

### Batch Execution

Create a batch script to run all experiments:

**Windows (batch_run_new_datasets.bat):**
```batch
@echo off
echo Running IMDB experiments...
for %%m in (mpnet qwen3 snowflake instructor jina_v3 bge e5) do (
    echo Running IMDB with %%m...
    python main.py --config experiments/exp_imdb_%%m.yaml
)

echo Running SST-2 experiments...
for %%m in (mpnet qwen3 snowflake instructor jina_v3 bge e5) do (
    echo Running SST-2 with %%m...
    python main.py --config experiments/exp_sst2_%%m.yaml
)

echo All experiments complete!
```

**Linux/Mac (batch_run_new_datasets.sh):**
```bash
#!/bin/bash
echo "Running IMDB experiments..."
for model in mpnet qwen3 snowflake instructor jina_v3 bge e5; do
    echo "Running IMDB with $model..."
    python main.py --config experiments/exp_imdb_$model.yaml
done

echo "Running SST-2 experiments..."
for model in mpnet qwen3 snowflake instructor jina_v3 bge e5; do
    echo "Running SST-2 with $model..."
    python main.py --config experiments/exp_sst2_$model.yaml
done

echo "All experiments complete!"
```

### Expected Runtime

- **IMDB:** ~5-10 minutes per model (long texts, avg 233 tokens)
- **SST-2:** ~2-5 minutes per model (short texts, avg 19 tokens)
- **Total:** ~1-2 hours for all 14 experiments

### Expected Outputs

Each experiment will generate:
- `results/raw/{dataset}_{model}_metrics.json` - Performance metrics
- `results/raw/{dataset}_{model}_predictions.csv` - Predictions for all samples

## Verification Steps

After running experiments, verify:

1. **All result files exist:**
   ```bash
   ls results/raw/imdb_*_metrics.json
   ls results/raw/sst2_*_metrics.json
   ```

2. **Results are reasonable:**
   - IMDB Macro-F1: Expected range 60-85%
   - SST-2 Macro-F1: Expected range 70-90%
   - Binary classification typically easier than multi-class

3. **Update results database:**
   - Ensure new experiments are added to consolidated results CSV
   - Verify all 63 experiments (49 old + 14 new) are present

## Next Steps

1. **Run Experiments:** Execute all 14 new experiments
2. **Update Analysis Scripts:** Modify analysis scripts to include new datasets
3. **Regenerate Visualizations:** Update heatmaps, tables, and plots
4. **Update Documentation:** Revise README and paper with new results
5. **Statistical Analysis:** Re-run Friedman and Nemenyi tests with 9 datasets

## Files Modified/Created

### Created Files:
- `scripts/analyze_dataset_coverage.py` - Coverage analysis script
- `scripts/evaluate_candidate_datasets.py` - Candidate evaluation script
- `docs/DATASET_INCLUSION_FRAMEWORK.md` - Decision framework
- `docs/DATASET_EXPANSION_SUMMARY.md` - This summary
- `experiments/exp_imdb_*.yaml` (7 files) - IMDB experiment configs
- `experiments/exp_sst2_*.yaml` (7 files) - SST-2 experiment configs

### Modified Files:
- `src/labels.py` - Added IMDB and SST-2 label definitions
- `src/data.py` - Added IMDB and SST-2 data loading

## Justification for TACL Submission

The dataset expansion strengthens the benchmark by:

1. **Addressing Reviewer Concerns:** Sentiment coverage was weak (1 dataset, financial only)
2. **Improving Diversity:** Added binary classification and long text representation
3. **Maintaining Focus:** Limited expansion (7 → 9 datasets) keeps benchmark manageable
4. **Adding Prestige:** SST-2 is a GLUE benchmark, widely recognized
5. **Clear Rationale:** Systematic evaluation framework provides defensible decisions

The expansion can be justified in the paper as:
> "To address limited sentiment diversity (1 dataset, financial domain only) and binary classification representation, we expanded the benchmark with IMDB (long-form movie reviews) and SST-2 (short-form movie reviews, GLUE benchmark). This expansion provides comprehensive sentiment coverage across text lengths while maintaining benchmark focus."

## Conclusion

Dataset expansion successfully implemented. The benchmark now provides:
- ✅ Strong sentiment coverage (3 datasets)
- ✅ Adequate binary classification (3 datasets)
- ✅ Improved text length diversity (2 long-text datasets)
- ✅ Systematic, defensible inclusion decisions
- ✅ All configurations standardized and ready to run

Total benchmark scope: 9 datasets × 7 models = 63 experiments
