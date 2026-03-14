"""Generate comprehensive publication-ready visualizations."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Better styling
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("=" * 80)
print("GENERATING BEAUTIFUL PUBLICATION-READY VISUALIZATIONS")
print("=" * 80)

# Setup
Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/plots").mkdir(parents=True, exist_ok=True)

# Load data
print("\n📂 Loading experiment results...")
metrics_files = list(Path("results/raw").glob("*_metrics.json"))
results = []

for f in metrics_files:
    with open(f, "r", encoding="utf-8") as fp:
        m = json.load(fp)
    
    exp_name = m.get("experiment_name", f.stem)
    
    # Extract model
    if "mpnet" in exp_name.lower():
        model = "MPNet"
    elif "jina_v5" in exp_name.lower():
        model = "Jina-v5"
    elif "qwen" in exp_name.lower():
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
        "accuracy": m["accuracy"] * 100,
        "macro_f1": m["macro_f1"] * 100,
        "weighted_f1": m["weighted_f1"] * 100,
        "precision": m.get("macro_precision", 0) * 100,
        "recall": m.get("macro_recall", 0) * 100,
        "confidence": m.get("mean_confidence", 0),
        "samples": m.get("num_samples", 0),
        "classes": len(set([k for k in m.get("classification_report", {}).keys() if k.isdigit()]))
    })

df = pd.DataFrame(results)
print(f"   ✅ Loaded {len(df)} experiments")

# ============================================================================
# PLOT 1: Comprehensive Model Comparison (4 metrics side by side)
# ============================================================================
print("\n📊 1. Creating comprehensive model comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

model_stats = df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'macro_f1': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std']
}).round(2)

metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
titles = ['Accuracy (%)', 'Macro F1 (%)', 'Precision (%)', 'Recall (%)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    means = model_stats[(metric, 'mean')].sort_values(ascending=False)
    stds = model_stats.loc[means.index, (metric, 'std')]
    
    bars = ax.barh(range(len(means)), means, xerr=stds, capsize=5, alpha=0.7)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index)
    ax.set_xlabel('Score (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (v, std) in enumerate(zip(means, stds)):
        ax.text(v + std + 1, i, f'{v:.1f}±{std:.1f}', va='center')

plt.tight_layout()
plt.savefig('results/plots/01_comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 01_comprehensive_model_comparison.png")

# ============================================================================
# PLOT 2: Performance Heatmap with Annotations
# ============================================================================
print("\n📊 2. Creating detailed performance heatmap...")
pivot = df.pivot_table(index='model', columns='dataset', values='macro_f1', aggfunc='mean')

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
            cbar_kws={'label': 'Macro F1 (%)'}, linewidths=0.5,
            vmin=30, vmax=80, ax=ax)
ax.set_title('Model Performance Heatmap (Macro F1 %)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Model', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/plots/02_performance_heatmap_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 02_performance_heatmap_detailed.png")

# ============================================================================
# PLOT 3: Dataset Difficulty Analysis (Multi-panel)
# ============================================================================
print("\n📊 3. Creating dataset difficulty analysis...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

dataset_stats = df.groupby(['dataset', 'classes']).agg({
    'accuracy': 'mean',
    'macro_f1': 'mean',
    'confidence': 'mean'
}).reset_index()

# Top left: Classes vs F1
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(dataset_stats['classes'], dataset_stats['macro_f1'], 
                     s=200, alpha=0.6, c=dataset_stats['macro_f1'], cmap='viridis')
for _, row in dataset_stats.iterrows():
    ax1.annotate(row['dataset'], (row['classes'], row['macro_f1']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax1.set_xlabel('Number of Classes', fontweight='bold')
ax1.set_ylabel('Macro F1 (%)', fontweight='bold')
ax1.set_title('Performance vs Dataset Complexity', fontweight='bold')
ax1.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Macro F1 (%)')

# Top right: Classes vs Confidence
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(dataset_stats['classes'], dataset_stats['confidence'], 
           s=200, alpha=0.6, c=dataset_stats['macro_f1'], cmap='viridis')
for _, row in dataset_stats.iterrows():
    ax2.annotate(row['dataset'], (row['classes'], row['confidence']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('Number of Classes', fontweight='bold')
ax2.set_ylabel('Mean Confidence', fontweight='bold')
ax2.set_title('Confidence vs Dataset Complexity', fontweight='bold')
ax2.grid(alpha=0.3)

# Bottom left: Performance distribution by dataset
ax3 = fig.add_subplot(gs[1, :])
df_sorted = df.sort_values('classes')
datasets = df_sorted['dataset'].unique()
positions = []
data_to_plot = []
for i, dataset in enumerate(datasets):
    data = df_sorted[df_sorted['dataset'] == dataset]['macro_f1'].values
    data_to_plot.append(data)
    positions.append(i)

bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                 showmeans=True, meanline=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
ax3.set_xticks(positions)
ax3.set_xticklabels(datasets, rotation=45, ha='right')
ax3.set_ylabel('Macro F1 (%)', fontweight='bold')
ax3.set_title('Performance Distribution Across Datasets', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('Dataset Difficulty & Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('results/plots/03_dataset_difficulty_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 03_dataset_difficulty_analysis.png")

# ============================================================================
# PLOT 4: Model Robustness (Performance Variability)
# ============================================================================
print("\n📊 4. Creating model robustness analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Box plot
model_order = df.groupby('model')['macro_f1'].mean().sort_values(ascending=False).index
data_to_plot = [df[df['model'] == model]['macro_f1'].values for model in model_order]

bp = axes[0].boxplot(data_to_plot, labels=model_order, patch_artist=True,
                     showmeans=True, meanline=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
axes[0].set_ylabel('Macro F1 (%)', fontweight='bold')
axes[0].set_title('Model Performance Distribution (Robustness)', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticklabels(model_order, rotation=15)

# Right: Consistency (mean vs std)
consistency = df.groupby('model').agg({
    'macro_f1': ['mean', 'std']
}).reset_index()
consistency.columns = ['model', 'mean', 'std']
consistency = consistency.sort_values('mean', ascending=False)

axes[1].scatter(consistency['std'], consistency['mean'], s=300, alpha=0.6)
for _, row in consistency.iterrows():
    axes[1].annotate(row['model'], (row['std'], row['mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
axes[1].set_xlabel('Standard Deviation', fontweight='bold')
axes[1].set_ylabel('Mean Macro F1 (%)', fontweight='bold')
axes[1].set_title('Model Consistency (Mean vs Variability)', fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axvline(consistency['std'].mean(), color='r', linestyle='--', alpha=0.5, label='Avg Std')
axes[1].legend()

plt.suptitle('Model Robustness Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/04_model_robustness.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 04_model_robustness.png")

# ============================================================================
# PLOT 5: Confidence Analysis (Multi-dimensional)
# ============================================================================
print("\n📊 5. Creating confidence analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Confidence Score Analysis', fontsize=16, fontweight='bold')

# Top left: Confidence by model
conf_by_model = df.groupby('model')['confidence'].mean().sort_values(ascending=False)
axes[0, 0].barh(range(len(conf_by_model)), conf_by_model.values, color='skyblue', alpha=0.7)
axes[0, 0].set_yticks(range(len(conf_by_model)))
axes[0, 0].set_yticklabels(conf_by_model.index)
axes[0, 0].set_xlabel('Mean Confidence', fontweight='bold')
axes[0, 0].set_title('Average Confidence by Model', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(conf_by_model.values):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Top right: Confidence by dataset
conf_by_dataset = df.groupby('dataset')['confidence'].mean().sort_values(ascending=False)
axes[0, 1].barh(range(len(conf_by_dataset)), conf_by_dataset.values, color='lightcoral', alpha=0.7)
axes[0, 1].set_yticks(range(len(conf_by_dataset)))
axes[0, 1].set_yticklabels(conf_by_dataset.index)
axes[0, 1].set_xlabel('Mean Confidence', fontweight='bold')
axes[0, 1].set_title('Average Confidence by Dataset', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(conf_by_dataset.values):
    axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

# Bottom left: Confidence vs Performance
axes[1, 0].scatter(df['confidence'], df['macro_f1'], alpha=0.5, s=100)
axes[1, 0].set_xlabel('Mean Confidence', fontweight='bold')
axes[1, 0].set_ylabel('Macro F1 (%)', fontweight='bold')
axes[1, 0].set_title('Confidence vs Performance Correlation', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Add correlation line
z = np.polyfit(df['confidence'], df['macro_f1'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['confidence'].min(), df['confidence'].max(), 100)
axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend (corr={df["confidence"].corr(df["macro_f1"]):.3f})')
axes[1, 0].legend()

# Bottom right: Confidence distribution
axes[1, 1].hist(df['confidence'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(df['confidence'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["confidence"].mean():.3f}')
axes[1, 1].axvline(df['confidence'].median(), color='b', linestyle='--', linewidth=2, label=f'Median: {df["confidence"].median():.3f}')
axes[1, 1].set_xlabel('Confidence Score', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Confidence Score Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/05_confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 05_confidence_analysis.png")

# ============================================================================
# PLOT 6: Radar Chart (Model Comparison)
# ============================================================================
print("\n📊 6. Creating radar chart for model comparison...")
from math import pi

categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

model_radar = df.groupby('model').agg({
    'accuracy': 'mean',
    'macro_f1': 'mean',
    'precision': 'mean',
    'recall': 'mean'
})

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

colors = plt.cm.Set2(np.linspace(0, 1, len(model_radar)))

for idx, (model, row) in enumerate(model_radar.iterrows()):
    values = row.values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 100)
ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('results/plots/06_model_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 06_model_radar_chart.png")

# ============================================================================
# PLOT 7: Winner Analysis
# ============================================================================
print("\n📊 7. Creating winner analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Best model per dataset
best_per_dataset = df.loc[df.groupby('dataset')['macro_f1'].idxmax()]
datasets = best_per_dataset['dataset'].values
models = best_per_dataset['model'].values
scores = best_per_dataset['macro_f1'].values

colors_map = {model: plt.cm.Set3(i) for i, model in enumerate(df['model'].unique())}
bar_colors = [colors_map[m] for m in models]

axes[0].barh(range(len(datasets)), scores, color=bar_colors, alpha=0.7)
axes[0].set_yticks(range(len(datasets)))
axes[0].set_yticklabels(datasets)
axes[0].set_xlabel('Best Macro F1 (%)', fontweight='bold')
axes[0].set_title('Best Model per Dataset', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

for i, (score, model) in enumerate(zip(scores, models)):
    axes[0].text(score + 1, i, f'{model} ({score:.1f}%)', va='center')

# Right: Win count per model
win_counts = best_per_dataset['model'].value_counts()
axes[1].bar(range(len(win_counts)), win_counts.values, color='gold', alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(win_counts)))
axes[1].set_xticklabels(win_counts.index)
axes[1].set_ylabel('Number of Wins', fontweight='bold')
axes[1].set_title('Model Win Count Across Datasets', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(win_counts.values):
    axes[1].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.suptitle('Winner Analysis: Best Models per Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/07_winner_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 07_winner_analysis.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL BEAUTIFUL PLOTS GENERATED!")
print("=" * 80)
print("\n📊 Generated 7 comprehensive visualizations:")
print("   1. Comprehensive Model Comparison (4 metrics)")
print("   2. Detailed Performance Heatmap")
print("   3. Dataset Difficulty Analysis (multi-panel)")
print("   4. Model Robustness Analysis")
print("   5. Confidence Analysis (4-panel)")
print("   6. Model Radar Chart")
print("   7. Winner Analysis")
print("\n📁 All plots saved to: results/plots/")
print("   High resolution (300 DPI) PNG format")
print("   Ready for publication! 🎉")