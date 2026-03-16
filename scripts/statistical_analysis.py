"""
Statistical analysis for benchmark results.

Implements:
- Friedman test for overall performance differences
- Nemenyi post-hoc test for pairwise comparisons
- Effect size computation
- Statistical power analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare
from pathlib import Path
import json


def load_results():
    """Load benchmark results from CSV."""
    df = pd.read_csv("results/MULTI_DATASET_RESULTS.csv")
    print(f"Loaded {len(df)} results")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    return df


def prepare_friedman_data(df):
    """
    Prepare data for Friedman test.
    
    Returns a matrix where:
    - Rows = datasets (blocks)
    - Columns = models (treatments)
    """
    # Create pivot table: datasets × models
    pivot = df.pivot_table(
        index='dataset',
        columns='model',
        values='macro_f1',
        aggfunc='mean'
    )
    
    print(f"\nFriedman test data shape: {pivot.shape}")
    print(f"Datasets (blocks): {pivot.shape[0]}")
    print(f"Models (treatments): {pivot.shape[1]}")
    
    # Check for missing values
    if pivot.isnull().any().any():
        print("\nWARNING: Missing values detected!")
        print(pivot.isnull().sum())
        print("\nDropping datasets with missing values...")
        pivot = pivot.dropna()
        print(f"New shape: {pivot.shape}")
    
    return pivot


def friedman_test(data):
    """
    Perform Friedman test.
    
    The Friedman test is a non-parametric alternative to repeated measures ANOVA.
    It tests whether k related samples (models) have different distributions.
    
    H0: All models have the same performance distribution
    H1: At least one model has a different performance distribution
    """
    print("\n" + "="*70)
    print("FRIEDMAN TEST")
    print("="*70)
    
    # Friedman test expects each row to be a block (dataset)
    # and each column to be a treatment (model)
    statistic, p_value = friedmanchisquare(*[data[col] for col in data.columns])
    
    n_datasets = len(data)
    n_models = len(data.columns)
    
    print(f"\nTest configuration:")
    print(f"  Number of datasets (blocks): {n_datasets}")
    print(f"  Number of models (treatments): {n_models}")
    print(f"  Total observations: {n_datasets * n_models}")
    
    print(f"\nResults:")
    print(f"  Friedman χ² statistic: {statistic:.4f}")
    print(f"  Degrees of freedom: {n_models - 1}")
    print(f"  P-value: {p_value:.6f}")
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"\n✓ SIGNIFICANT (p < {alpha})")
        print("  Reject H0: Models have significantly different performance")
    else:
        print(f"\n✗ NOT SIGNIFICANT (p >= {alpha})")
        print("  Fail to reject H0: No significant difference between models")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'n_datasets': n_datasets,
        'n_models': n_models,
        'df': n_models - 1
    }


def compute_effect_size(data):
    """
    Compute effect size for Friedman test.
    
    Uses Kendall's W (coefficient of concordance):
    W = χ²_F / (n * (k - 1))
    
    where:
    - χ²_F is the Friedman statistic
    - n is the number of blocks (datasets)
    - k is the number of treatments (models)
    
    Interpretation:
    - W = 0: No agreement
    - W = 1: Perfect agreement
    - W < 0.3: Weak effect
    - 0.3 ≤ W < 0.5: Moderate effect
    - W ≥ 0.5: Strong effect
    """
    print("\n" + "="*70)
    print("EFFECT SIZE (Kendall's W)")
    print("="*70)
    
    n = len(data)  # number of datasets (blocks)
    k = len(data.columns)  # number of models (treatments)
    
    # Compute Friedman statistic
    statistic, _ = friedmanchisquare(*[data[col] for col in data.columns])
    
    # Kendall's W
    W = statistic / (n * (k - 1))
    
    print(f"\nKendall's W: {W:.4f}")
    
    # Interpretation
    if W < 0.3:
        interpretation = "Weak effect"
    elif W < 0.5:
        interpretation = "Moderate effect"
    else:
        interpretation = "Strong effect"
    
    print(f"Interpretation: {interpretation}")
    
    # Additional context
    print(f"\nContext:")
    print(f"  W ranges from 0 (no agreement) to 1 (perfect agreement)")
    print(f"  W = {W:.4f} indicates {interpretation.lower()} in model rankings across datasets")
    
    return {
        'kendalls_w': W,
        'interpretation': interpretation
    }


def compute_average_ranks(data):
    """
    Compute average ranks for each model across datasets.
    
    Lower rank = better performance.
    """
    print("\n" + "="*70)
    print("AVERAGE RANKS")
    print("="*70)
    
    # Rank models for each dataset (1 = best, k = worst)
    # Note: We use ascending=False because higher macro_f1 is better
    ranks = data.rank(axis=1, ascending=False)
    
    # Compute average rank for each model
    avg_ranks = ranks.mean(axis=0).sort_values()
    
    print("\nAverage ranks (lower is better):")
    for i, (model, rank) in enumerate(avg_ranks.items(), 1):
        print(f"  {i}. {model:20s} {rank:.2f}")
    
    return avg_ranks


def critical_distance_nemenyi(n_datasets, n_models, alpha=0.05):
    """
    Compute critical distance for Nemenyi post-hoc test.
    
    CD = q_α * sqrt(k(k+1) / (6n))
    
    where:
    - q_α is the critical value from Studentized range distribution
    - k is the number of models
    - n is the number of datasets
    
    Two models are significantly different if their average rank difference > CD.
    """
    # Critical values for Nemenyi test (two-tailed, α=0.05)
    # Source: Demšar (2006), Table 1
    # These are q_α values for different numbers of treatments
    q_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    if n_models not in q_values:
        raise ValueError(f"Critical value not available for {n_models} models")
    
    q_alpha = q_values[n_models]
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    
    return cd, q_alpha


def nemenyi_test(data, avg_ranks, alpha=0.05):
    """
    Perform Nemenyi post-hoc test for pairwise comparisons.
    
    The Nemenyi test is used after a significant Friedman test to determine
    which specific pairs of models differ significantly.
    """
    print("\n" + "="*70)
    print("NEMENYI POST-HOC TEST")
    print("="*70)
    
    n_datasets = len(data)
    n_models = len(data.columns)
    
    # Compute critical distance
    cd, q_alpha = critical_distance_nemenyi(n_datasets, n_models, alpha)
    
    print(f"\nTest configuration:")
    print(f"  Number of datasets: {n_datasets}")
    print(f"  Number of models: {n_models}")
    print(f"  Significance level (α): {alpha}")
    print(f"  Critical value (q_α): {q_alpha:.3f}")
    print(f"  Critical distance (CD): {cd:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Two models are significantly different if |rank_i - rank_j| > {cd:.4f}")
    
    # Perform pairwise comparisons
    models = avg_ranks.index.tolist()
    comparisons = []
    
    print(f"\nPairwise comparisons:")
    print(f"{'Model 1':<20s} {'Model 2':<20s} {'Rank Diff':>10s} {'Significant':>12s}")
    print("-" * 70)
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]
            rank_diff = abs(avg_ranks[model1] - avg_ranks[model2])
            is_significant = rank_diff > cd
            
            comparisons.append({
                'model1': model1,
                'model2': model2,
                'rank_diff': rank_diff,
                'significant': is_significant
            })
            
            sig_marker = "✓ YES" if is_significant else "✗ NO"
            print(f"{model1:<20s} {model2:<20s} {rank_diff:>10.4f} {sig_marker:>12s}")
    
    # Count significant differences
    n_significant = sum(1 for c in comparisons if c['significant'])
    n_total = len(comparisons)
    
    print(f"\nSummary:")
    print(f"  Significant differences: {n_significant}/{n_total}")
    
    return {
        'critical_distance': cd,
        'q_alpha': q_alpha,
        'comparisons': comparisons,
        'n_significant': n_significant,
        'n_total': n_total
    }


def detect_cliques(avg_ranks, cd):
    """
    Detect cliques (groups of models with no significant differences).
    
    A clique is a maximal set of models where no two models differ by more than CD.
    """
    print("\n" + "="*70)
    print("CLIQUE DETECTION")
    print("="*70)
    
    models = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    
    # Sort by rank
    sorted_indices = np.argsort(ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = [ranks[i] for i in sorted_indices]
    
    # Find cliques using a greedy approach
    cliques = []
    used = set()
    
    for i in range(len(sorted_models)):
        if sorted_models[i] in used:
            continue
        
        # Start a new clique
        clique = [sorted_models[i]]
        clique_ranks = [sorted_ranks[i]]
        
        # Add models that don't differ significantly from any model in the clique
        for j in range(i + 1, len(sorted_models)):
            if sorted_models[j] in used:
                continue
            
            # Check if model j is within CD of all models in current clique
            if all(abs(sorted_ranks[j] - r) <= cd for r in clique_ranks):
                clique.append(sorted_models[j])
                clique_ranks.append(sorted_ranks[j])
        
        cliques.append(clique)
        used.update(clique)
    
    print(f"\nFound {len(cliques)} clique(s):")
    for i, clique in enumerate(cliques, 1):
        print(f"\nClique {i} ({len(clique)} models):")
        for model in clique:
            print(f"  - {model} (rank: {avg_ranks[model]:.2f})")
    
    return cliques


def compute_statistical_power(data, effect_size_w, alpha=0.05):
    """
    Compute statistical power for Friedman test.
    
    Statistical power is the probability of correctly rejecting H0 when it is false.
    
    For Friedman test, we use the approximation:
    - Power depends on: sample size (n), number of treatments (k), effect size (W), and α
    - We compute the non-centrality parameter and use chi-square distribution
    
    Power interpretation:
    - Power < 0.5: Very low power, results unreliable
    - 0.5 ≤ Power < 0.8: Low to moderate power
    - Power ≥ 0.8: Adequate power (standard threshold)
    - Power ≥ 0.9: High power
    """
    print("\n" + "="*70)
    print("STATISTICAL POWER ANALYSIS")
    print("="*70)
    
    n = len(data)  # number of datasets (blocks)
    k = len(data.columns)  # number of models (treatments)
    
    # Non-centrality parameter for Friedman test
    # λ = n * (k - 1) * W
    ncp = n * (k - 1) * effect_size_w
    
    # Critical value for chi-square distribution
    df = k - 1
    critical_value = stats.chi2.ppf(1 - alpha, df)
    
    # Power = P(χ² > critical_value | λ = ncp)
    # This is 1 - CDF of non-central chi-square at critical value
    power = 1 - stats.ncx2.cdf(critical_value, df, ncp)
    
    print(f"\nPower analysis configuration:")
    print(f"  Number of datasets (n): {n}")
    print(f"  Number of models (k): {k}")
    print(f"  Effect size (Kendall's W): {effect_size_w:.4f}")
    print(f"  Significance level (α): {alpha}")
    print(f"  Degrees of freedom: {df}")
    print(f"  Non-centrality parameter (λ): {ncp:.4f}")
    print(f"  Critical value (χ²): {critical_value:.4f}")
    
    print(f"\nStatistical power: {power:.4f} ({power*100:.2f}%)")
    
    # Interpretation
    if power < 0.5:
        interpretation = "Very low power - results unreliable"
        recommendation = "Increase sample size significantly or effect size is too small to detect"
    elif power < 0.8:
        interpretation = "Low to moderate power - results may be unreliable"
        recommendation = "Consider increasing sample size for more reliable conclusions"
    elif power < 0.9:
        interpretation = "Adequate power - results are reasonably reliable"
        recommendation = "Sample size is sufficient for detecting the observed effect"
    else:
        interpretation = "High power - results are highly reliable"
        recommendation = "Sample size is more than sufficient"
    
    print(f"Interpretation: {interpretation}")
    print(f"Recommendation: {recommendation}")
    
    # Sample size recommendations
    print(f"\nSample size analysis:")
    print(f"  Current: {n} datasets")
    
    # Compute required sample size for 80% power
    target_power = 0.8
    required_n = n
    for test_n in range(1, 100):
        test_ncp = test_n * (k - 1) * effect_size_w
        test_power = 1 - stats.ncx2.cdf(critical_value, df, test_ncp)
        if test_power >= target_power:
            required_n = test_n
            break
    
    if required_n <= n:
        print(f"  Required for 80% power: {required_n} datasets ✓ (sufficient)")
    else:
        print(f"  Required for 80% power: {required_n} datasets ✗ (need {required_n - n} more)")
    
    # Compute required sample size for 90% power
    target_power = 0.9
    required_n_90 = n
    for test_n in range(1, 100):
        test_ncp = test_n * (k - 1) * effect_size_w
        test_power = 1 - stats.ncx2.cdf(critical_value, df, test_ncp)
        if test_power >= target_power:
            required_n_90 = test_n
            break
    
    if required_n_90 <= n:
        print(f"  Required for 90% power: {required_n_90} datasets ✓ (sufficient)")
    else:
        print(f"  Required for 90% power: {required_n_90} datasets ✗ (need {required_n_90 - n} more)")
    
    return {
        'power': power,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'n_datasets': n,
        'n_models': k,
        'effect_size_w': effect_size_w,
        'ncp': ncp,
        'required_n_80': required_n,
        'required_n_90': required_n_90
    }


def verify_sample_sizes(df):
    """
    Verify that sample sizes are sufficient and consistent across experiments.
    """
    print("\n" + "="*70)
    print("SAMPLE SIZE VERIFICATION")
    print("="*70)
    
    # Check sample sizes by dataset
    sample_sizes = df.groupby('dataset')['samples'].agg(['mean', 'std', 'min', 'max'])
    
    print("\nSample sizes by dataset:")
    print(sample_sizes.to_string())
    
    # Check for inconsistencies
    inconsistent = sample_sizes[sample_sizes['std'] > 0]
    if len(inconsistent) > 0:
        print("\n⚠️  WARNING: Inconsistent sample sizes detected!")
        print(inconsistent.to_string())
    else:
        print("\n✓ All sample sizes are consistent within each dataset")
    
    # Overall statistics
    print(f"\nOverall sample size statistics:")
    print(f"  Mean: {df['samples'].mean():.0f}")
    print(f"  Median: {df['samples'].median():.0f}")
    print(f"  Min: {df['samples'].min():.0f}")
    print(f"  Max: {df['samples'].max():.0f}")
    
    # Check if sample sizes are adequate
    min_recommended = 1000
    datasets_below = df[df['samples'] < min_recommended]['dataset'].unique()
    
    if len(datasets_below) > 0:
        print(f"\n⚠️  Datasets with sample size < {min_recommended}:")
        for ds in datasets_below:
            size = df[df['dataset'] == ds]['samples'].iloc[0]
            print(f"  - {ds}: {size}")
    else:
        print(f"\n✓ All datasets have sample size ≥ {min_recommended}")
    
    return {
        'sample_sizes': sample_sizes.to_dict(),
        'mean': float(df['samples'].mean()),
        'median': float(df['samples'].median()),
        'min': int(df['samples'].min()),
        'max': int(df['samples'].max()),
        'inconsistent_datasets': inconsistent.index.tolist() if len(inconsistent) > 0 else []
    }


def main():
    """Run complete statistical analysis."""
    print("="*70)
    print("STATISTICAL ANALYSIS FOR BENCHMARK RESULTS")
    print("="*70)
    
    # Load data
    df = load_results()
    
    # Prepare data for Friedman test
    data = prepare_friedman_data(df)
    
    # 1. Friedman test
    friedman_results = friedman_test(data)
    
    # 2. Effect size
    effect_size_results = compute_effect_size(data)
    
    # 3. Average ranks
    avg_ranks = compute_average_ranks(data)
    
    # 4. Nemenyi post-hoc test (only if Friedman is significant)
    if friedman_results['p_value'] < 0.05:
        nemenyi_results = nemenyi_test(data, avg_ranks)
        
        # 5. Clique detection
        cliques = detect_cliques(avg_ranks, nemenyi_results['critical_distance'])
    else:
        print("\n" + "="*70)
        print("SKIPPING POST-HOC TESTS")
        print("="*70)
        print("\nFriedman test was not significant, so post-hoc tests are not needed.")
        nemenyi_results = None
        cliques = None
    
    # 6. Statistical power analysis
    power_results = compute_statistical_power(
        data, 
        effect_size_results['kendalls_w']
    )
    
    # 7. Sample size verification
    sample_size_results = verify_sample_sizes(df)
    
    # Save results
    results = {
        'friedman': friedman_results,
        'effect_size': effect_size_results,
        'average_ranks': avg_ranks.to_dict(),
        'nemenyi': nemenyi_results,
        'cliques': cliques,
        'power_analysis': power_results,
        'sample_sizes': sample_size_results
    }
    
    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "statistical_tests.json", "w") as f:
        # Convert numpy types and other non-serializable types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return str(obj)
        
        json.dump(results, f, indent=2, default=convert)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir / 'statistical_tests.json'}")


if __name__ == "__main__":
    main()
