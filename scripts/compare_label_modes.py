"""Compare different label modes (L1, L2, L3, etc.) across models and datasets.

This script analyzes results from experiments using different label formulations
and generates comparison tables and plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_all_results():
    """Load all experiment results from results/raw/."""
    results_dir = Path("results/raw")
    all_results = []
    
    for file in results_dir.glob("*_metrics.json"):
        with open(file) as f:
            data = json.load(f)
            all_results.append(data)
    
    return pd.DataFrame(all_results)


def identify_label_mode(exp_name, label_mode_field=None):
    """Identify label mode from experiment name or field."""
    if label_mode_field:
        return label_mode_field
    
    # From experiment name
    if "_l2" in exp_name:
        return "l2"
    elif "_l3" in exp_name:
        return "l3"
    elif "name_only" in exp_name or exp_name.count("_") <= 3:
        return "name_only"
    elif "description" in exp_name:
        return "description"
    else:
        return "unknown"


def compare_label_modes():
    """Generate label mode comparison analysis."""
    
    print("\n" + "="*70)
    print("LABEL MODE COMPARISON ANALYSIS")
    print("="*70 + "\n")
    
    # Load results
    df = load_all_results()
    
    if df.empty:
        print("❌ No results found in results/raw/")
        print("   Run experiments first!")
        return
    
    # Add label mode if not present
    if "label_mode" not in df.columns:
        df["label_mode"] = df["experiment_name"].apply(identify_label_mode)
    else:
        df["label_mode"] = df.apply(
            lambda row: identify_label_mode(row["experiment_name"], row.get("label_mode")),
            axis=1
        )
    
    # Extract model name
    if "biencoder" in df.columns:
        df["model"] = df["biencoder"].apply(lambda x: x.split("/")[-1] if isinstance(x, str) else "unknown")
    
    print(f"📊 Loaded {len(df)} experiment results\n")
    
    # 1. Overall comparison by label mode
    print("=" * 70)
    print("1. AVERAGE ACCURACY BY LABEL MODE")
    print("=" * 70)
    mode_comparison = df.groupby("label_mode")["accuracy"].agg(["mean", "std", "count"])
    mode_comparison = mode_comparison.sort_values("mean", ascending=False)
    print(mode_comparison.to_string())
    print()
    
    # 2. Per-dataset comparison
    print("=" * 70)
    print("2. ACCURACY BY DATASET AND LABEL MODE")
    print("=" * 70)
    dataset_comparison = df.pivot_table(
        index="dataset",
        columns="label_mode",
        values="accuracy",
        aggfunc="mean"
    )
    print(dataset_comparison.to_string())
    print()
    
    # 3. Per-model comparison
    if "model" in df.columns:
        print("=" * 70)
        print("3. ACCURACY BY MODEL AND LABEL MODE")
        print("=" * 70)
        model_comparison = df.pivot_table(
            index="model",
            columns="label_mode",
            values="accuracy",
            aggfunc="mean"
        )
        print(model_comparison.to_string())
        print()
    
    # 4. Best label mode per dataset
    print("=" * 70)
    print("4. BEST LABEL MODE PER DATASET")
    print("=" * 70)
    best_per_dataset = df.loc[df.groupby("dataset")["accuracy"].idxmax()]
    print(best_per_dataset[["dataset", "label_mode", "accuracy", "experiment_name"]].to_string(index=False))
    print()
    
    # 5. Statistical summary
    print("=" * 70)
    print("5. LABEL MODE STATISTICS")
    print("=" * 70)
    for mode in df["label_mode"].unique():
        mode_data = df[df["label_mode"] == mode]["accuracy"]
        print(f"\n{mode.upper()}:")
        print(f"  Mean:   {mode_data.mean():.4f}")
        print(f"  Median: {mode_data.median():.4f}")
        print(f"  Std:    {mode_data.std():.4f}")
        print(f"  Min:    {mode_data.min():.4f}")
        print(f"  Max:    {mode_data.max():.4f}")
        print(f"  Count:  {len(mode_data)}")
    
    # Save detailed comparison
    output_file = Path("results/label_mode_comparison.csv")
    comparison_df = df[[
        "dataset", "label_mode", "model", "accuracy", "f1_weighted",
        "experiment_name"
    ]].sort_values(["dataset", "label_mode", "accuracy"], ascending=[True, True, False])
    
    comparison_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    compare_label_modes()