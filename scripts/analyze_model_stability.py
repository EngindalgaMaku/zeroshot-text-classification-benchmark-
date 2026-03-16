"""
Model Stability Analysis

Computes stability metrics for each model across datasets:
- Coefficient of variation (CV) for consistency measurement
- Mean and standard deviation of Macro-F1 scores
- Stability ranking table

Validates Requirements 9.1, 9.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_results():
    """Load benchmark results from CSV."""
    df = pd.read_csv("results/MULTI_DATASET_RESULTS.csv")
    print(f"Loaded {len(df)} results")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    return df


def compute_stability_metrics(df):
    """
    Compute stability metrics for each model.
    
    Stability is measured using coefficient of variation (CV):
    CV = (std / mean) * 100
    
    Lower CV indicates more stable/consistent performance across datasets.
    
    Returns:
        DataFrame with columns: model, mean_f1, std_f1, cv, rank
    """
    print("\n" + "="*70)
    print("COMPUTING STABILITY METRICS")
    print("="*70)
    
    # Group by model and compute statistics
    stability = df.groupby('model')['macro_f1'].agg([
        ('mean_f1', 'mean'),
        ('std_f1', 'std'),
        ('min_f1', 'min'),
        ('max_f1', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Compute coefficient of variation (CV)
    # CV = (std / mean) * 100
    # Lower CV = more stable/consistent
    stability['cv'] = (stability['std_f1'] / stability['mean_f1']) * 100
    
    # Rank by stability (lower CV = better rank)
    stability['stability_rank'] = stability['cv'].rank(method='min').astype(int)
    
    # Rank by performance (higher mean = better rank)
    stability['performance_rank'] = stability['mean_f1'].rank(ascending=False, method='min').astype(int)
    
    # Sort by stability (most stable first)
    stability = stability.sort_values('cv')
    
    return stability


def print_stability_table(stability):
    """Print formatted stability ranking table."""
    print("\n" + "="*70)
    print("MODEL STABILITY RANKING")
    print("="*70)
    print("\nRanked by Coefficient of Variation (lower = more stable)")
    print("-" * 70)
    print(f"{'Rank':<6} {'Model':<20} {'Mean F1':<10} {'Std F1':<10} {'CV (%)':<10} {'Range':<15}")
    print("-" * 70)
    
    for idx, row in stability.iterrows():
        rank = row['stability_rank']
        model = row['model']
        mean_f1 = row['mean_f1']
        std_f1 = row['std_f1']
        cv = row['cv']
        f1_range = f"{row['min_f1']:.1f}-{row['max_f1']:.1f}"
        
        print(f"{rank:<6} {model:<20} {mean_f1:<10.2f} {std_f1:<10.2f} {cv:<10.2f} {f1_range:<15}")
    
    print("-" * 70)
    print(f"\nInterpretation:")
    print(f"  CV < 10%:  Very stable performance")
    print(f"  CV 10-20%: Moderately stable performance")
    print(f"  CV > 20%:  Variable performance across datasets")


def analyze_stability_performance_tradeoff(stability):
    """
    Analyze the relationship between stability and performance.
    
    Identifies models with:
    - High performance + high stability (ideal)
    - High performance + low stability (inconsistent)
    - Low performance + high stability (consistently weak)
    - Low performance + low stability (worst case)
    """
    print("\n" + "="*70)
    print("STABILITY-PERFORMANCE TRADE-OFF ANALYSIS")
    print("="*70)
    
    # Define thresholds
    median_f1 = stability['mean_f1'].median()
    median_cv = stability['cv'].median()
    
    print(f"\nThresholds:")
    print(f"  Median Mean F1: {median_f1:.2f}")
    print(f"  Median CV: {median_cv:.2f}")
    
    # Categorize models
    stability['category'] = 'Unknown'
    
    for idx, row in stability.iterrows():
        if row['mean_f1'] >= median_f1 and row['cv'] <= median_cv:
            stability.at[idx, 'category'] = 'High Perf + High Stability (IDEAL)'
        elif row['mean_f1'] >= median_f1 and row['cv'] > median_cv:
            stability.at[idx, 'category'] = 'High Perf + Low Stability'
        elif row['mean_f1'] < median_f1 and row['cv'] <= median_cv:
            stability.at[idx, 'category'] = 'Low Perf + High Stability'
        else:
            stability.at[idx, 'category'] = 'Low Perf + Low Stability'
    
    # Print categorization
    print("\nModel Categorization:")
    print("-" * 70)
    
    for category in ['High Perf + High Stability (IDEAL)', 
                     'High Perf + Low Stability',
                     'Low Perf + High Stability',
                     'Low Perf + Low Stability']:
        models_in_cat = stability[stability['category'] == category]
        if len(models_in_cat) > 0:
            print(f"\n{category}:")
            for _, row in models_in_cat.iterrows():
                print(f"  - {row['model']:<20} (F1: {row['mean_f1']:.2f}, CV: {row['cv']:.2f}%)")
    
    return stability


def identify_best_tradeoffs(stability):
    """
    Identify models with best stability-performance trade-offs.
    
    Uses a composite score that balances performance and stability:
    Score = mean_f1 - (cv * weight)
    
    Higher score = better trade-off
    """
    print("\n" + "="*70)
    print("BEST STABILITY-PERFORMANCE TRADE-OFFS")
    print("="*70)
    
    # Compute composite score
    # Weight CV penalty (higher weight = more emphasis on stability)
    cv_weight = 0.5  # Adjust this to change stability vs performance emphasis
    
    stability['composite_score'] = stability['mean_f1'] - (stability['cv'] * cv_weight)
    stability['composite_rank'] = stability['composite_score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by composite score
    best_tradeoffs = stability.sort_values('composite_score', ascending=False)
    
    print(f"\nComposite Score = Mean F1 - (CV × {cv_weight})")
    print("Higher score = better balance of performance and stability")
    print("-" * 70)
    print(f"{'Rank':<6} {'Model':<20} {'Mean F1':<10} {'CV (%)':<10} {'Score':<10}")
    print("-" * 70)
    
    for idx, row in best_tradeoffs.iterrows():
        rank = row['composite_rank']
        model = row['model']
        mean_f1 = row['mean_f1']
        cv = row['cv']
        score = row['composite_score']
        
        print(f"{rank:<6} {model:<20} {mean_f1:<10.2f} {cv:<10.2f} {score:<10.2f}")
    
    print("-" * 70)
    
    # Identify top 3
    top_3 = best_tradeoffs.head(3)
    print(f"\n🏆 Top 3 Models (Best Trade-offs):")
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. {row['model']}: F1={row['mean_f1']:.2f}, CV={row['cv']:.2f}%, Score={row['composite_score']:.2f}")
    
    return best_tradeoffs


def analyze_performance_stability_correlation(stability):
    """
    Analyze whether high-performing models are also stable.
    
    Computes correlation between mean performance and stability (CV).
    """
    print("\n" + "="*70)
    print("PERFORMANCE-STABILITY CORRELATION")
    print("="*70)
    
    # Compute correlation
    correlation = stability['mean_f1'].corr(stability['cv'])
    
    print(f"\nPearson correlation between Mean F1 and CV: {correlation:.4f}")
    
    # Interpretation
    if correlation < -0.3:
        interpretation = "Strong negative correlation: High performers are MORE stable"
    elif correlation < -0.1:
        interpretation = "Weak negative correlation: High performers are slightly more stable"
    elif correlation < 0.1:
        interpretation = "No correlation: Performance and stability are independent"
    elif correlation < 0.3:
        interpretation = "Weak positive correlation: High performers are slightly less stable"
    else:
        interpretation = "Strong positive correlation: High performers are LESS stable"
    
    print(f"Interpretation: {interpretation}")
    
    # Additional analysis: Compare top vs bottom performers
    top_performers = stability.nlargest(3, 'mean_f1')
    bottom_performers = stability.nsmallest(3, 'mean_f1')
    
    print(f"\nTop 3 Performers:")
    print(f"  Mean CV: {top_performers['cv'].mean():.2f}%")
    print(f"  Models: {', '.join(top_performers['model'].tolist())}")
    
    print(f"\nBottom 3 Performers:")
    print(f"  Mean CV: {bottom_performers['cv'].mean():.2f}%")
    print(f"  Models: {', '.join(bottom_performers['model'].tolist())}")
    
    cv_diff = top_performers['cv'].mean() - bottom_performers['cv'].mean()
    if cv_diff < 0:
        print(f"\n✓ Top performers are {abs(cv_diff):.2f}% more stable (lower CV)")
    else:
        print(f"\n✗ Top performers are {cv_diff:.2f}% less stable (higher CV)")
    
    return correlation


def save_results(stability, correlation):
    """Save stability analysis results."""
    output_dir = Path("results/stability_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stability table as CSV
    csv_path = output_dir / "model_stability_ranking.csv"
    stability.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Saved stability ranking: {csv_path}")
    
    # Save as formatted text table
    txt_path = output_dir / "model_stability_ranking.txt"
    with open(txt_path, 'w') as f:
        f.write("MODEL STABILITY RANKING\n")
        f.write("="*70 + "\n\n")
        f.write("Ranked by Coefficient of Variation (lower = more stable)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Model':<20} {'Mean F1':<10} {'Std F1':<10} {'CV (%)':<10} {'Range':<15}\n")
        f.write("-" * 70 + "\n")
        
        for _, row in stability.sort_values('cv').iterrows():
            rank = row['stability_rank']
            model = row['model']
            mean_f1 = row['mean_f1']
            std_f1 = row['std_f1']
            cv = row['cv']
            f1_range = f"{row['min_f1']:.1f}-{row['max_f1']:.1f}"
            
            f.write(f"{rank:<6} {model:<20} {mean_f1:<10.2f} {std_f1:<10.2f} {cv:<10.2f} {f1_range:<15}\n")
        
        f.write("-" * 70 + "\n")
    
    print(f"✅ Saved formatted table: {txt_path}")
    
    # Save summary statistics as JSON
    summary = {
        'correlation_f1_cv': float(correlation),
        'models': []
    }
    
    for _, row in stability.iterrows():
        summary['models'].append({
            'model': row['model'],
            'mean_f1': float(row['mean_f1']),
            'std_f1': float(row['std_f1']),
            'cv': float(row['cv']),
            'min_f1': float(row['min_f1']),
            'max_f1': float(row['max_f1']),
            'stability_rank': int(row['stability_rank']),
            'performance_rank': int(row['performance_rank']),
            'composite_rank': int(row['composite_rank']),
            'category': row['category']
        })
    
    json_path = output_dir / "stability_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved JSON summary: {json_path}")


def main():
    """Run complete stability analysis."""
    print("="*70)
    print("MODEL STABILITY ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_results()
    
    # Compute stability metrics
    stability = compute_stability_metrics(df)
    
    # Print stability ranking table
    print_stability_table(stability)
    
    # Analyze stability-performance trade-off
    stability = analyze_stability_performance_tradeoff(stability)
    
    # Identify best trade-offs
    stability = identify_best_tradeoffs(stability)
    
    # Analyze correlation
    correlation = analyze_performance_stability_correlation(stability)
    
    # Save results
    save_results(stability, correlation)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"  • Most stable model: {stability.iloc[0]['model']} (CV: {stability.iloc[0]['cv']:.2f}%)")
    print(f"  • Least stable model: {stability.iloc[-1]['model']} (CV: {stability.iloc[-1]['cv']:.2f}%)")
    print(f"  • Best trade-off: {stability.sort_values('composite_score', ascending=False).iloc[0]['model']}")
    print(f"  • Performance-stability correlation: {correlation:.4f}")


if __name__ == "__main__":
    main()
