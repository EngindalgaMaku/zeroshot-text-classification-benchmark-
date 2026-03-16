# Statistical Analysis Results

## Overview

This directory contains comprehensive statistical analysis results for the benchmark comparing 7 sentence embedding models across 8-9 datasets.

## Files

### 1. `statistical_tests.json`
Machine-readable JSON file containing all statistical test results:
- Friedman test results (χ² statistic, p-value, degrees of freedom)
- Effect size (Kendall's W)
- Average ranks for each model
- Nemenyi post-hoc test results (critical distance, pairwise comparisons)
- Model cliques (groups with no significant differences)
- Statistical power analysis
- Sample size verification

### 2. `POWER_ANALYSIS.md`
Human-readable comprehensive documentation including:
- Detailed methodology explanations
- Statistical power analysis and interpretation
- Sample size requirements and verification
- Post-hoc analysis results
- Model cliques and pairwise comparisons
- Recommendations for future work
- References

## Key Findings

### Overall Significance
- **Friedman Test:** χ² = 16.45, p = 0.012 (significant at α = 0.05)
- **Conclusion:** Models have significantly different performance across datasets

### Effect Size
- **Kendall's W:** 0.343 (moderate effect)
- **Interpretation:** Moderate agreement in model rankings across datasets

### Statistical Power
- **Power:** 87.96% (adequate, exceeds 80% threshold)
- **Conclusion:** Sample size is sufficient for reliable statistical conclusions

### Model Rankings (Average Ranks)
1. Qwen3 (2.00) ⭐ Best
2. INSTRUCTOR (2.88)
3. E5-large (3.38)
4. Jina v5 (4.50)
5. MPNet (4.75)
6. BGE-M3 (5.25)
7. Nomic-MoE (5.25)

### Significant Pairwise Differences
- **Qwen3 vs BGE-M3:** Significant (rank diff = 3.25 > CD = 3.19)
- **Qwen3 vs Nomic-MoE:** Significant (rank diff = 3.25 > CD = 3.19)

### Model Cliques
**Clique 1 (Top performers):** Qwen3, INSTRUCTOR, E5-large, Jina v5, MPNet
**Clique 2 (Lower performers):** BGE-M3, Nomic-MoE

## Usage

### Running the Analysis
```bash
python scripts/statistical_analysis.py
```

This will:
1. Load results from `results/MULTI_DATASET_RESULTS.csv`
2. Perform Friedman test
3. Compute effect size (Kendall's W)
4. Calculate average ranks
5. Perform Nemenyi post-hoc test (if Friedman is significant)
6. Detect model cliques
7. Compute statistical power
8. Verify sample sizes
9. Save results to `statistical_tests.json`

### Interpreting Results

**Friedman Test:**
- Tests if models have different performance distributions
- Significant result (p < 0.05) means models differ significantly

**Effect Size (Kendall's W):**
- Measures strength of agreement in rankings
- 0 = no agreement, 1 = perfect agreement
- < 0.3 = weak, 0.3-0.5 = moderate, ≥ 0.5 = strong

**Nemenyi Test:**
- Post-hoc test for pairwise comparisons
- Two models differ significantly if rank difference > critical distance (CD)

**Statistical Power:**
- Probability of detecting a true effect
- ≥ 0.8 is considered adequate
- ≥ 0.9 is considered high

## Requirements

The analysis requires:
- Python 3.7+
- pandas
- numpy
- scipy

Install dependencies:
```bash
pip install pandas numpy scipy
```

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.
- Nemenyi, P. (1963). *Distribution-free multiple comparisons*. Princeton University, PhD thesis.

---

*Last updated: 2024*
