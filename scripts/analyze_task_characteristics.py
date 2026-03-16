"""
Task Characteristics Correlation Analysis

Correlates task characteristics (num_classes, avg_text_length, label_similarity)
with model performance (Macro-F1) to identify strongest predictors.

**Validates: Requirements 8.1, 8.2, 8.5**
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json


def load_data():
    """Load task characteristics and benchmark results."""
    chars_path = Path("results/task_characteristics/task_characteristics.csv")
    results_path = Path("results/MULTI_DATASET_RESULTS.csv")

    if not chars_path.exists():
        raise FileNotFoundError(
            f"Task characteristics not found at {chars_path}. "
            "Please run compute_task_characteristics.py first."
        )

    chars_df = pd.read_csv(chars_path)
    results_df = pd.read_csv(results_path)

    print(f"Loaded characteristics for {len(chars_df)} datasets")
    print(f"Loaded {len(results_df)} benchmark results")
    print(f"Models: {sorted(results_df['model'].unique())}")
    return chars_df, results_df


def merge_data(chars_df, results_df):
    """Merge characteristics with per-model results."""
    # Drop num_classes from results if it exists (we use the one from chars_df)
    results_clean = results_df.drop(columns=["num_classes"], errors="ignore")
    merged = results_clean.merge(chars_df, on="dataset", how="inner")
    print(f"\nMerged dataset: {len(merged)} rows "
          f"({merged['dataset'].nunique()} datasets × {merged['model'].nunique()} models)")
    return merged


def compute_correlations(merged_df):
    """
    Compute Pearson and Spearman correlations between task characteristics
    and Macro-F1 scores.

    Returns a DataFrame with correlation results for each characteristic.
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: TASK CHARACTERISTICS vs MACRO-F1")
    print("=" * 70)

    characteristics = ["num_classes", "avg_text_length", "label_similarity"]
    char_labels = {
        "num_classes": "Number of Classes",
        "avg_text_length": "Average Text Length",
        "label_similarity": "Label Semantic Similarity",
    }

    results = []

    for char in characteristics:
        x = merged_df[char].values
        y = merged_df["macro_f1"].values

        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(x, y)

        # Spearman correlation (monotonic, rank-based)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        results.append({
            "characteristic": char,
            "label": char_labels[char],
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_r),
            "abs_spearman": abs(spearman_r),
        })

        print(f"\n{char_labels[char]}:")
        print(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4f})")
        print(f"  Spearman r = {spearman_r:+.4f}  (p = {spearman_p:.4f})")

        sig_p = "✓ significant (p<0.05)" if pearson_p < 0.05 else "✗ not significant"
        sig_s = "✓ significant (p<0.05)" if spearman_p < 0.05 else "✗ not significant"
        print(f"  Pearson:  {sig_p}")
        print(f"  Spearman: {sig_s}")

    corr_df = pd.DataFrame(results)
    return corr_df


def compute_per_model_correlations(merged_df):
    """
    Compute correlations separately for each model to see if patterns
    are consistent across models.
    """
    print("\n" + "=" * 70)
    print("PER-MODEL CORRELATION ANALYSIS")
    print("=" * 70)

    characteristics = ["num_classes", "avg_text_length", "label_similarity"]
    models = sorted(merged_df["model"].unique())
    rows = []

    for model in models:
        model_df = merged_df[merged_df["model"] == model]
        row = {"model": model}
        for char in characteristics:
            x = model_df[char].values
            y = model_df["macro_f1"].values
            if len(x) < 3:
                row[f"{char}_r"] = np.nan
                row[f"{char}_p"] = np.nan
            else:
                r, p = stats.spearmanr(x, y)
                row[f"{char}_r"] = r
                row[f"{char}_p"] = p
        rows.append(row)

    per_model_df = pd.DataFrame(rows)

    print("\nSpearman correlations per model:")
    print(f"{'Model':<20} {'num_classes':>12} {'text_length':>12} {'label_sim':>12}")
    print("-" * 60)
    for _, row in per_model_df.iterrows():
        nc = f"{row['num_classes_r']:+.3f}" if not np.isnan(row['num_classes_r']) else "  N/A"
        tl = f"{row['avg_text_length_r']:+.3f}" if not np.isnan(row['avg_text_length_r']) else "  N/A"
        ls = f"{row['label_similarity_r']:+.3f}" if not np.isnan(row['label_similarity_r']) else "  N/A"
        print(f"{row['model']:<20} {nc:>12} {tl:>12} {ls:>12}")

    return per_model_df


def identify_strongest_predictors(corr_df):
    """
    Identify which task characteristics most strongly predict model performance.
    """
    print("\n" + "=" * 70)
    print("STRONGEST PREDICTORS OF MODEL PERFORMANCE")
    print("=" * 70)

    # Rank by absolute Pearson correlation
    ranked = corr_df.sort_values("abs_pearson", ascending=False)

    print("\nRanked by absolute Pearson correlation (strongest predictor first):")
    print("-" * 60)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        sig = "✓" if row["pearson_p"] < 0.05 else "✗"
        direction = "negative" if row["pearson_r"] < 0 else "positive"
        print(f"  {i}. {row['label']:<30} r={row['pearson_r']:+.4f} ({direction}) {sig}")

    best = ranked.iloc[0]
    print(f"\n→ Strongest predictor: {best['label']}")
    print(f"  r = {best['pearson_r']:+.4f}, p = {best['pearson_p']:.4f}")

    if best["pearson_r"] < 0:
        print(f"  Interpretation: More classes → lower Macro-F1 (harder tasks)")
    else:
        print(f"  Interpretation: Higher {best['label']} → higher Macro-F1")

    return ranked


def compute_dataset_level_correlations(chars_df, results_df):
    """
    Compute correlations at the dataset level (using mean Macro-F1 per dataset).
    This gives a cleaner signal by averaging out model-specific variance.
    """
    print("\n" + "=" * 70)
    print("DATASET-LEVEL CORRELATION (mean F1 per dataset)")
    print("=" * 70)

    mean_f1 = results_df.groupby("dataset")["macro_f1"].mean().reset_index()
    mean_f1.columns = ["dataset", "mean_macro_f1"]

    dataset_df = chars_df.merge(mean_f1, on="dataset")

    characteristics = ["num_classes", "avg_text_length", "label_similarity"]
    char_labels = {
        "num_classes": "Number of Classes",
        "avg_text_length": "Average Text Length",
        "label_similarity": "Label Semantic Similarity",
    }

    rows = []
    for char in characteristics:
        x = dataset_df[char].values
        y = dataset_df["mean_macro_f1"].values
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        rows.append({
            "characteristic": char,
            "label": char_labels[char],
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_r": r_s,
            "spearman_p": p_s,
        })
        print(f"\n{char_labels[char]}:")
        print(f"  Pearson  r = {r_p:+.4f}  (p = {p_p:.4f})")
        print(f"  Spearman r = {r_s:+.4f}  (p = {p_s:.4f})")

    return pd.DataFrame(rows), dataset_df


def save_results(corr_df, per_model_df, dataset_corr_df, dataset_df):
    """Save correlation analysis results."""
    output_dir = Path("results/task_characteristics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall correlations
    corr_path = output_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False, float_format="%.6f")
    print(f"\n✅ Saved overall correlations: {corr_path}")

    # Per-model correlations
    per_model_path = output_dir / "per_model_correlations.csv"
    per_model_df.to_csv(per_model_path, index=False, float_format="%.6f")
    print(f"✅ Saved per-model correlations: {per_model_path}")

    # Dataset-level correlations
    dataset_corr_path = output_dir / "dataset_level_correlations.csv"
    dataset_corr_df.to_csv(dataset_corr_path, index=False, float_format="%.6f")
    print(f"✅ Saved dataset-level correlations: {dataset_corr_path}")

    # Dataset characteristics with mean F1 (used by visualization)
    dataset_df_path = output_dir / "dataset_characteristics_with_f1.csv"
    dataset_df.to_csv(dataset_df_path, index=False, float_format="%.6f")
    print(f"✅ Saved dataset characteristics with F1: {dataset_df_path}")

    # JSON summary
    summary = {
        "overall_correlations": corr_df[
            ["characteristic", "label", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
        ].to_dict(orient="records"),
        "dataset_level_correlations": dataset_corr_df[
            ["characteristic", "label", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
        ].to_dict(orient="records"),
    }
    json_path = output_dir / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved JSON summary: {json_path}")


def main():
    """Run complete task characteristics correlation analysis."""
    print("=" * 70)
    print("TASK CHARACTERISTICS CORRELATION ANALYSIS")
    print("=" * 70)

    chars_df, results_df = load_data()
    merged_df = merge_data(chars_df, results_df)

    # Overall correlations (all models × all datasets)
    corr_df = compute_correlations(merged_df)

    # Per-model breakdown
    per_model_df = compute_per_model_correlations(merged_df)

    # Dataset-level (mean F1 per dataset)
    dataset_corr_df, dataset_df = compute_dataset_level_correlations(chars_df, results_df)

    # Identify strongest predictors
    ranked = identify_strongest_predictors(corr_df)

    # Save all results
    save_results(corr_df, per_model_df, dataset_corr_df, dataset_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    best = ranked.iloc[0]
    print(f"\nKey finding: Strongest predictor of Macro-F1 is '{best['label']}' "
          f"(r={best['pearson_r']:+.4f})")


if __name__ == "__main__":
    main()
