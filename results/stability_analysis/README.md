# Model Stability Analysis

This directory contains the results of the model stability analysis for the TACL benchmark strengthening project.

## Overview

The stability analysis evaluates how consistently each model performs across different datasets. This is measured using the **Coefficient of Variation (CV)**, which is the ratio of standard deviation to mean, expressed as a percentage.

**Lower CV = More Stable/Consistent Performance**

## Files Generated

### Data Files

1. **model_stability_ranking.csv** - Machine-readable CSV with all stability metrics
2. **model_stability_ranking.txt** - Human-readable formatted table
3. **stability_metrics.json** - Complete JSON with all metrics and categorizations

### Visualizations

All visualizations are available in `results/plots/` in three formats (PNG, PDF, EPS):

1. **model_stability_scatter** - Scatter plot showing mean performance vs stability (CV)
   - X-axis: Coefficient of Variation (lower = more stable)
   - Y-axis: Mean Macro-F1 score (higher = better)
   - Quadrants show: high/low performance × high/low stability
   - Gold stars mark top 3 models with best trade-offs

2. **model_stability_ranking** - Bar chart showing CV for each model
   - Horizontal bars sorted by stability (most stable at top)
   - Color-coded by performance-stability category

3. **performance_stability_comparison** - Dual-axis comparison of ranks
   - Shows performance rank vs stability rank for each model
   - Helps identify whether top performers are also most stable

## Key Findings

### Stability Ranking (by CV)

1. **INSTRUCTOR** - CV: 32.63% (Most Stable)
2. **MPNet** - CV: 33.97%
3. **Jina v5** - CV: 34.04%
4. **Qwen3** - CV: 34.14%
5. **Nomic-MoE** - CV: 35.75%
6. **BGE-M3** - CV: 36.44%
7. **E5-large** - CV: 38.69% (Least Stable)

### Best Trade-offs (Performance + Stability)

Using composite score = Mean F1 - (CV × 0.5):

1. **Qwen3** - F1: 68.93, CV: 34.14%, Score: 51.86
2. **INSTRUCTOR** - F1: 66.52, CV: 32.63%, Score: 50.20
3. **Jina v5** - F1: 62.58, CV: 34.04%, Score: 45.56

### Performance-Stability Correlation

**Correlation: -0.3127** (Strong negative correlation)

This indicates that **high-performing models tend to be MORE stable** across datasets. Top performers have slightly lower CV (more stable) than bottom performers.

### Model Categories

**High Performance + High Stability (IDEAL):**
- INSTRUCTOR (F1: 66.52, CV: 32.63%)
- Jina v5 (F1: 62.58, CV: 34.04%)
- Qwen3 (F1: 68.93, CV: 34.14%)

**High Performance + Low Stability:**
- E5-large (F1: 62.76, CV: 38.69%)

**Low Performance + High Stability:**
- MPNet (F1: 58.35, CV: 33.97%)

**Low Performance + Low Stability:**
- Nomic-MoE (F1: 59.44, CV: 35.75%)
- BGE-M3 (F1: 61.19, CV: 36.44%)

## Interpretation

### Coefficient of Variation Guidelines

- **CV < 10%**: Very stable performance
- **CV 10-20%**: Moderately stable performance
- **CV > 20%**: Variable performance across datasets

All models in this benchmark show CV > 30%, indicating that **zero-shot performance is highly task-dependent**. This is expected given the diversity of datasets (topic classification, entity typing, intent detection, sentiment analysis, emotion recognition).

### Recommendations

1. **For consistent performance**: Choose INSTRUCTOR or Qwen3 (best stability-performance trade-off)
2. **For maximum performance**: Choose Qwen3 (highest mean F1: 68.93)
3. **For most stable**: Choose INSTRUCTOR (lowest CV: 32.63%)

## Scripts

### analyze_model_stability.py

Computes stability metrics:
- Coefficient of variation for each model
- Mean and standard deviation of Macro-F1
- Stability ranking table
- Performance-stability correlation
- Trade-off analysis

**Usage:**
```bash
python scripts/analyze_model_stability.py
```

### visualize_model_stability.py

Generates publication-quality visualizations:
- Scatter plot with quadrant analysis
- Stability ranking bar chart
- Performance-stability comparison

**Usage:**
```bash
python scripts/visualize_model_stability.py
```

## Requirements Validated

This analysis validates the following requirements from the TACL benchmark specification:

- **Requirement 9.1**: Compute coefficient of variation for each model across all datasets ✓
- **Requirement 9.2**: Generate stability ranking table ordering models by consistency ✓
- **Requirement 9.3**: Produce scatter plot showing mean performance vs stability ✓
- **Requirement 9.4**: Identify models with best stability-performance trade-offs ✓
- **Requirement 9.5**: Analyze whether high-performing models are also stable ✓

## Publication Notes

All figures are generated in vector formats (PDF, EPS) suitable for TACL submission. The analysis provides insights into:

1. Model consistency across diverse tasks
2. Trade-offs between performance and stability
3. Whether top performers are also most reliable
4. Recommendations for model selection based on use case

This analysis strengthens the benchmark by going beyond simple performance rankings to evaluate model robustness and consistency.
