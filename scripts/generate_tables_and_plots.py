"""Generate publication-ready tables and plots from experiment results."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("GENERATING TABLES AND PLOTS")
print("=" * 70)

# Create output directories
Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/plots").mkdir(parents=True, exist_ok=True)

# 1. Load all results
print("\n1. Loading results...")
metrics_files = list(Path("results/raw").glob("*_metrics.json"))

results = []
for f in metrics_files:
    with open(f, "r", encoding="utf-8") as fp:
        m = json.load(fp)
    
    # Extract model name
    exp_name = m.get("experiment_name", f.stem)
    if "mpnet" in exp_name.lower():
        model = "MPNet"
    elif "jina_v5" in exp_name.lower():
        model = "Jina v5"
    elif "qwen3" in exp_name.lower() or "qwen" in exp_name.lower():
        model = "Qwen3"
    elif "bge" in exp_name.lower():
        model = "BGE-M3"
    elif "e5" in exp_name.lower():
        model = "E5-large"
    else:
        model = "Unknown"
    
    results.append({
        "experiment": exp_name,
        "dataset": m.get("dataset_name", "N/A"),
        "model": model,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "macro_precision": m.get("macro_precision", 0.0),
        "macro_recall": m.get("macro_recall", 0.0),
        "mean_confidence": m.get("mean_confidence", 0.0),
        "num_samples": m.get("num_samples", 0),
        "num_classes": len(set([k for k in m.get("classification_report", {}).keys() if k.isdigit()]))
    })

results_df = pd.DataFrame(results)
print(f"   Loaded {len(results_df)} experiments")

# 2. Main results table
print("\n2. Creating main results table...")
main_table = results_df[["dataset", "model", "num_classes", "accuracy", "macro_f1", "weighted_f1"]].copy()
for col in ["accuracy", "macro_f1", "weighted_f1"]:
    main_table[col] = (main_table[col] * 100).round(2)
main_table = main_table.sort_values("macro_f1", ascending=False)

main_table.to_csv("results/tables/main_results.csv", index=False)
print("   ✅ main_results.csv")

# 3. Model comparison
print("\n3. Creating model comparison...")
model_df = results_df.groupby("model").agg({
    "accuracy": "mean", "macro_f1": "mean", "weighted_f1": "mean"
}).reset_index().sort_values("macro_f1", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_df))
width = 0.25
ax.bar(x - width, model_df["accuracy"], width, label='Accuracy')
ax.bar(x, model_df["macro_f1"], width, label='Macro F1')
ax.bar(x + width, model_df["weighted_f1"], width, label='Weighted F1')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Average Performance by Model')
ax.set_xticks(x)
ax.set_xticklabels(model_df["model"])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ model_comparison.png")

# 4. Dataset difficulty
print("\n4. Creating difficulty analysis...")
dataset_df = results_df.groupby(["dataset", "num_classes"]).agg({
    "accuracy": "mean", "macro_f1": "mean"
}).reset_index().sort_values("num_classes")

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(dataset_df["num_classes"], dataset_df["macro_f1"], s=100, alpha=0.6)
for idx, row in dataset_df.iterrows():
    ax.annotate(row["dataset"], (row["num_classes"], row["macro_f1"]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel("Number of Classes")
ax.set_ylabel("Average Macro F1")
ax.set_title("Performance vs Dataset Difficulty")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/difficulty_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ difficulty_analysis.png")

# 5. Performance heatmap
print("\n5. Creating performance heatmap...")
pivot_df = results_df.pivot_table(index="model", columns="dataset", values="macro_f1", aggfunc="mean")

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Macro F1'})
plt.title('Macro F1 Score Heatmap (Model × Dataset)')
plt.xlabel('Dataset')
plt.ylabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/plots/performance_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ performance_heatmap.png")

# 6. Confidence analysis
print("\n6. Creating confidence analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

conf_by_model = results_df.groupby("model")["mean_confidence"].mean().sort_values(ascending=False)
ax1.barh(range(len(conf_by_model)), conf_by_model.values)
ax1.set_yticks(range(len(conf_by_model)))
ax1.set_yticklabels(conf_by_model.index)
ax1.set_xlabel("Mean Confidence Score")
ax1.set_title("Average Confidence by Model")
ax1.set_xlim(0, 1)

conf_by_dataset = results_df.groupby("dataset")["mean_confidence"].mean().sort_values(ascending=False)
ax2.barh(range(len(conf_by_dataset)), conf_by_dataset.values)
ax2.set_yticks(range(len(conf_by_dataset)))
ax2.set_yticklabels(conf_by_dataset.index)
ax2.set_xlabel("Mean Confidence Score")
ax2.set_title("Average Confidence by Dataset")
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("results/plots/confidence_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ confidence_analysis.png")

# 7. Summary stats
print("\n7. Creating summary statistics...")
summary = results_df[["accuracy", "macro_f1", "weighted_f1"]].describe()
summary = (summary * 100).round(2)
summary.to_csv("results/tables/summary_statistics.csv")
print("   ✅ summary_statistics.csv")

# 8. Top 10
print("\n8. Creating top 10 table...")
top10 = results_df.nlargest(10, "macro_f1")[
    ["experiment", "dataset", "model", "num_classes", "accuracy", "macro_f1", "weighted_f1"]
].copy()
for col in ["accuracy", "macro_f1", "weighted_f1"]:
    top10[col] = (top10[col] * 100).round(2)
top10.to_csv("results/tables/top10_results.csv", index=False)
print("   ✅ top10_results.csv")

# 9. Model consistency
print("\n9. Creating model consistency analysis...")
consistency = results_df.groupby("model").agg({
    "macro_f1": ["mean", "std", "min", "max"]
}).round(4)
consistency.columns = ['_'.join(col) for col in consistency.columns]
consistency = consistency.sort_values("macro_f1_mean", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
models = consistency.index
means = consistency["macro_f1_mean"]
stds = consistency["macro_f1_std"]
ax.bar(range(len(models)), means, yerr=stds, capsize=5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models)
ax.set_ylabel("Macro F1")
ax.set_title("Model Performance Consistency (Mean ± Std)")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/model_consistency.png", dpi=300, bbox_inches="tight")
plt.close()

consistency.to_csv("results/tables/model_consistency.csv")
print("   ✅ model_consistency.png & .csv")

# 10. Excel export
print("\n10. Exporting to Excel...")
try:
    with pd.ExcelWriter("results/tables/all_results.xlsx") as writer:
        results_df.to_excel(writer, sheet_name="All Results", index=False)
        main_table.to_excel(writer, sheet_name="Main Table", index=False)
        top10.to_excel(writer, sheet_name="Top 10", index=False)
        summary.to_excel(writer, sheet_name="Summary Stats")
        model_df.to_excel(writer, sheet_name="Model Comparison", index=False)
        consistency.to_excel(writer, sheet_name="Model Consistency")
    print("   ✅ all_results.xlsx")
except ImportError:
    print("   ⚠️  openpyxl not installed - skipping Excel export")

print("\n" + "=" * 70)
print("✅ ALL DONE!")
print("=" * 70)
print(f"\n📁 Tables: results/tables/")
print(f"📊 Plots:  results/plots/")
print(f"\n{len(list(Path('results/tables').glob('*')))} tables generated")
print(f"{len(list(Path('results/plots').glob('*.png')))} plots generated")