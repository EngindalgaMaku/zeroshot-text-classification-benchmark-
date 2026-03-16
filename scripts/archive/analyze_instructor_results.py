"""Analyze INSTRUCTOR results across all 6 datasets."""

import json
import pandas as pd
from pathlib import Path

# Load all INSTRUCTOR results
results_dir = Path("results/raw")
instructor_files = list(results_dir.glob("*_instructor_metrics.json"))

data = []
for file in instructor_files:
    with open(file, encoding='utf-8') as f:
        metrics = json.load(f)
    
    dataset_name = file.stem.replace("_instructor_metrics", "").replace("_", " ").title()
    
    data.append({
        "Dataset": dataset_name,
        "Samples": metrics["num_samples"],
        "Classes": len([k for k in metrics["classification_report"].keys() if k.isdigit()]),
        "Accuracy": metrics["accuracy"] * 100,
        "Macro F1": metrics["macro_f1"] * 100,
        "Precision": metrics.get("macro_precision", 0) * 100,
        "Recall": metrics.get("macro_recall", 0) * 100,
        "Avg Confidence": metrics.get("mean_confidence", 0) * 100,
    })

df = pd.DataFrame(data)
df = df.sort_values("Macro F1", ascending=False)

print("=" * 100)
print("INSTRUCTOR PERFORMANCE ACROSS 6 DATASETS")
print("=" * 100)
print(df.to_string(index=False))
print()

# Summary statistics
print("=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)
print(f"Average Macro F1 across all datasets: {df['Macro F1'].mean():.2f}%")
print(f"Best dataset: {df.iloc[0]['Dataset']} ({df.iloc[0]['Macro F1']:.2f}% F1)")
print(f"Hardest dataset: {df.iloc[-1]['Dataset']} ({df.iloc[-1]['Macro F1']:.2f}% F1)")
print()

# Performance by difficulty (number of classes)
print("=" * 100)
print("PERFORMANCE VS DATASET DIFFICULTY")
print("=" * 100)
df_sorted = df.sort_values("Classes")
for _, row in df_sorted.iterrows():
    print(f"{row['Dataset']:30s} | Classes: {row['Classes']:3.0f} | F1: {row['Macro F1']:6.2f}%")
print()

# Save results
df.to_csv("results/INSTRUCTOR_ALL_DATASETS.csv", index=False)
print("✅ Saved to results/INSTRUCTOR_ALL_DATASETS.csv")