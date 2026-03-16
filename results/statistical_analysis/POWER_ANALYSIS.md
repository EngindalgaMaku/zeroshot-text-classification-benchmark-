# Statistical Power Analysis

## Overview

This document provides a comprehensive statistical power analysis for the benchmark comparing 7 sentence embedding models across multiple datasets using Macro-F1 scores.

## Methodology

### Friedman Test

The Friedman test is a non-parametric alternative to repeated measures ANOVA, used to test whether k related samples (models) have different distributions across multiple blocks (datasets).

**Hypotheses:**
- H₀: All models have the same performance distribution
- H₁: At least one model has a different performance distribution

**Test Configuration:**
- Number of datasets (blocks): 8
- Number of models (treatments): 7
- Total observations: 56
- Significance level (α): 0.05

**Results:**
- Friedman χ² statistic: 16.4464
- Degrees of freedom: 6
- P-value: 0.011548
- **Conclusion:** Significant (p < 0.05) - Models have significantly different performance

### Effect Size (Kendall's W)

Kendall's W (coefficient of concordance) measures the strength of agreement in rankings across datasets.

**Formula:** W = χ²_F / (n × (k - 1))

where:
- χ²_F is the Friedman statistic
- n is the number of datasets (blocks)
- k is the number of models (treatments)

**Result:** W = 0.3426

**Interpretation:**
- W ranges from 0 (no agreement) to 1 (perfect agreement)
- W = 0.3426 indicates a **moderate effect** in model rankings across datasets
- This suggests that while models show consistent relative performance patterns, there is still substantial variation across datasets

**Effect Size Guidelines:**
- W < 0.3: Weak effect
- 0.3 ≤ W < 0.5: Moderate effect
- W ≥ 0.5: Strong effect

## Statistical Power Analysis

Statistical power is the probability of correctly rejecting H₀ when it is false (i.e., detecting a true effect).

### Power Calculation

**Configuration:**
- Number of datasets (n): 8
- Number of models (k): 7
- Effect size (Kendall's W): 0.3426
- Significance level (α): 0.05
- Non-centrality parameter (λ): 16.4464

**Result:** Power = 0.8796 (87.96%)

**Interpretation:** Adequate power - results are reasonably reliable

### Power Guidelines

- Power < 0.5: Very low power, results unreliable
- 0.5 ≤ Power < 0.8: Low to moderate power
- **Power ≥ 0.8: Adequate power (standard threshold)** ✓
- Power ≥ 0.9: High power

### Sample Size Requirements

**For 80% Power:**
- Required: 7 datasets
- Current: 8 datasets
- **Status:** ✓ Sufficient

**For 90% Power:**
- Required: 9 datasets
- Current: 8 datasets
- **Status:** Need 1 more dataset

### Conclusion

The current benchmark with 8 datasets provides **adequate statistical power (87.96%)** to detect the observed moderate effect size (W = 0.3426). This exceeds the standard 80% power threshold, indicating that:

1. The sample size is sufficient for reliable statistical conclusions
2. The probability of Type II error (false negative) is acceptably low (12.04%)
3. The significant Friedman test result (p = 0.012) is reliable and not due to insufficient power

To achieve 90% power (high power), adding just 1 more dataset would be beneficial but not strictly necessary given the current adequate power level.

## Sample Size Verification

### Per-Dataset Sample Sizes

| Dataset | Sample Size | Status |
|---------|-------------|--------|
| 20 Newsgroups | 2000 | ✓ |
| AG News | 1000 | ✓ |
| Banking77 | 1000 | ✓ |
| DBpedia-14 | 1000 | ✓ |
| GoEmotions | 1000 | ✓ |
| IMDB | 1000 | ✓ |
| SST-2 | 872 | ⚠️ Below 1000 |
| Twitter Financial | 1000 | ✓ |
| Yahoo Answers | 1000 | ✓ |

### Overall Statistics

- Mean: 1082 samples
- Median: 1000 samples
- Min: 872 samples (SST-2)
- Max: 2000 samples (20 Newsgroups)

### Assessment

All datasets have sample sizes ≥ 872, which is sufficient for reliable Macro-F1 estimation. The SST-2 dataset has a slightly smaller sample size (872) due to the dataset's natural size constraints, but this is still adequate for the analysis.

The sample sizes are consistent within each dataset (no variation across models), ensuring fair comparisons.

## Post-Hoc Analysis (Nemenyi Test)

### Critical Distance

**Formula:** CD = q_α × √(k(k+1) / (6n))

where:
- q_α is the critical value from Studentized range distribution
- k is the number of models
- n is the number of datasets

**Result:** CD = 3.1853

**Interpretation:** Two models are significantly different if their average rank difference > 3.1853

### Significant Pairwise Differences

Out of 21 pairwise comparisons, **2 were significant**:

1. **Qwen3 vs BGE-M3** (rank difference: 3.25)
2. **Qwen3 vs Nomic-MoE** (rank difference: 3.25)

### Model Cliques

Models are grouped into cliques where no two models within a clique differ significantly:

**Clique 1 (Top performers):**
- Qwen3 (rank: 2.00)
- INSTRUCTOR (rank: 2.88)
- E5-large (rank: 3.38)
- Jina v5 (rank: 4.50)
- MPNet (rank: 4.75)

**Clique 2 (Lower performers):**
- BGE-M3 (rank: 5.25)
- Nomic-MoE (rank: 5.25)

### Interpretation

The analysis reveals that:
1. Qwen3 is the top-performing model and significantly outperforms BGE-M3 and Nomic-MoE
2. The top 5 models (Qwen3, INSTRUCTOR, E5-large, Jina v5, MPNet) form a statistically homogeneous group
3. BGE-M3 and Nomic-MoE form a separate group with significantly lower performance than Qwen3

## Recommendations

1. **Sample Size:** The current 8 datasets provide adequate statistical power (87.96%). No immediate action required, though adding 1 more dataset would achieve 90% power.

2. **Per-Dataset Samples:** All datasets have sufficient sample sizes (≥ 872). The variation in sample sizes (872-2000) is acceptable and reflects dataset availability.

3. **Statistical Validity:** The significant Friedman test (p = 0.012) with adequate power confirms that model performance differences are real and not due to chance or insufficient sample size.

4. **Model Comparisons:** Focus on the distinction between the top-performing group (Clique 1) and lower-performing models (Clique 2), particularly highlighting Qwen3's superior performance.

5. **Future Work:** If expanding the benchmark, prioritize adding diverse datasets that complement existing task types to maintain or increase the moderate effect size.

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.
- Nemenyi, P. (1963). *Distribution-free multiple comparisons*. Princeton University, PhD thesis.

---

*Generated by statistical_analysis.py*
