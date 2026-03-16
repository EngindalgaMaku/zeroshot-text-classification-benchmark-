"""
Model Stability Visualization

Generates publication-quality visualizations for model stability analysis:
- Scatter plot: mean performance vs stability (CV)
- Quadrant analysis showing high/low performance and stability regions
- Identification and annotation of models with best trade-offs

Validates Requirements 9.3, 9.4, 9.5
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_stability_data():
    """Load stability metrics from JSON."""
    json_path = Path("results/stability_analysis/stability_metrics.json")
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Stability metrics not found at {json_path}. "
            "Please run analyze_model_stability.py first."
        )
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['models'])
    print(f"Loaded stability data for {len(df)} models")
    
    return df, data['correlation_f1_cv']


def compute_pareto_frontier(df):
    """
    Compute the Pareto frontier of models not dominated in both dimensions.
    
    A model is Pareto-optimal if no other model has both higher mean_f1 AND lower cv.
    Returns the frontier points sorted by cv for plotting a trade-off curve.
    """
    points = df[['cv', 'mean_f1']].values
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower cv AND higher mean_f1
            if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                    is_pareto[i] = False
                    break
    pareto_df = df[is_pareto].sort_values('cv')
    return pareto_df


def create_stability_scatter_plot(df):
    """
    Create publication-quality scatter plot: mean performance vs stability (CV).
    
    - X-axis: Coefficient of Variation (CV) - lower is more stable
    - Y-axis: Mean Macro-F1 - higher is better performance
    - Quadrant lines at median values
    - Model-specific colors consistent with MODEL_COLORS palette
    - Pareto frontier trade-off curve
    - No title; top/right spines removed
    """
    print("\n" + "="*70)
    print("CREATING STABILITY SCATTER PLOT")
    print("="*70)

    MODEL_COLORS = {
        "BGE-M3":     "#1f77b4",
        "E5-large":   "#ff7f0e",
        "INSTRUCTOR": "#2ca02c",
        "Jina v5":    "#d62728",
        "MPNet":      "#9467bd",
        "Nomic-MoE":  "#8c564b",
        "Qwen3":      "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    # Calculate medians for quadrant lines
    median_f1 = df['mean_f1'].median()
    median_cv = df['cv'].median()

    print(f"\nQuadrant thresholds:")
    print(f"  Median Mean F1: {median_f1:.2f}")
    print(f"  Median CV: {median_cv:.2f}")

    # Add quadrant lines at median values
    ax.axhline(y=median_f1, color='gray', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
    ax.axvline(x=median_cv, color='gray', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)

    # Add quadrant labels
    x_min, x_max = df['cv'].min(), df['cv'].max()
    y_min, y_max = df['mean_f1'].min(), df['mean_f1'].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    pad = 0.04

    quadrant_kwargs = dict(fontsize=9, color='gray', alpha=0.55, style='italic')
    ax.text(x_min + x_range * pad, y_max - y_range * pad,
            'IDEAL\nHigh Perf · High Stability',
            ha='left', va='top', **quadrant_kwargs)
    ax.text(x_max - x_range * pad, y_max - y_range * pad,
            'High Perf\nLow Stability',
            ha='right', va='top', **quadrant_kwargs)
    ax.text(x_min + x_range * pad, y_min + y_range * pad,
            'Low Perf\nHigh Stability',
            ha='left', va='bottom', **quadrant_kwargs)
    ax.text(x_max - x_range * pad, y_min + y_range * pad,
            'Low Perf\nLow Stability',
            ha='right', va='bottom', **quadrant_kwargs)

    # Pareto frontier trade-off curve
    pareto_df = compute_pareto_frontier(df)
    print(f"\nPareto-optimal models ({len(pareto_df)}):")
    for _, row in pareto_df.iterrows():
        print(f"  {row['model']}: F1={row['mean_f1']:.2f}, CV={row['cv']:.2f}%")

    if len(pareto_df) >= 2:
        ax.plot(
            pareto_df['cv'], pareto_df['mean_f1'],
            color='dimgray', linestyle='-', linewidth=1.5,
            alpha=0.5, zorder=2, label='Pareto frontier'
        )

    # Plot each model with its own color
    for _, row in df.iterrows():
        color = MODEL_COLORS.get(row['model'], '#333333')
        ax.scatter(
            row['cv'], row['mean_f1'],
            c=color, s=180,
            edgecolors='black', linewidth=1.2,
            zorder=3, label=row['model']
        )

    # Annotate model names
    for _, row in df.iterrows():
        ax.annotate(
            row['model'],
            xy=(row['cv'], row['mean_f1']),
            xytext=(6, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.75),
            zorder=4
        )

    # Axis labels
    ax.set_xlabel('Coefficient of Variation (%) ← More Stable', fontsize=12)
    ax.set_ylabel('Mean Macro-F1 Score', fontsize=12)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend (models + Pareto frontier)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right', frameon=True,
              framealpha=0.9, fontsize=9)

    plt.tight_layout()

    return fig, ax


def save_figure(fig, filename_base):
    """Save figure in multiple formats for publication."""
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    png_path = output_dir / f"{filename_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved PNG: {png_path}")
    
    # Save as PDF (vector format for publication)
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✅ Saved PDF: {pdf_path}")
    
    # Save as EPS (alternative vector format)
    eps_path = output_dir / f"{filename_base}.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"✅ Saved EPS: {eps_path}")


def create_stability_bar_chart(df):
    """
    Create bar chart showing CV for each model.
    
    Provides a clear ranking of model stability.
    """
    print("\n" + "="*70)
    print("CREATING STABILITY BAR CHART")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by CV (most stable first)
    df_sorted = df.sort_values('cv')
    
    # Color bars by category
    colors = []
    for category in df_sorted['category']:
        if 'IDEAL' in category:
            colors.append('#2E86AB')  # Blue
        elif 'High Perf + Low Stability' in category:
            colors.append('#F18F01')  # Orange
        elif 'Low Perf + High Stability' in category:
            colors.append('#6A994E')  # Green
        else:
            colors.append('#C73E1D')  # Red
    
    # Create bars
    bars = ax.barh(
        range(len(df_sorted)),
        df_sorted['cv'],
        color=colors,
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    
    # Add value labels
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(
            row['cv'] + 0.5,
            i,
            f"{row['cv']:.2f}%",
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Y-axis: model names
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['model'], fontsize=11)
    
    # X-axis
    ax.set_xlabel('Coefficient of Variation (%) ← More Stable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Add vertical line at median
    median_cv = df['cv'].median()
    ax.axvline(x=median_cv, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(
        median_cv,
        len(df_sorted) - 0.5,
        f' Median: {median_cv:.2f}%',
        fontsize=9,
        color='gray',
        va='top'
    )
    
    # Grid
    ax.grid(True, axis='x', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Invert x-axis so more stable (lower CV) is on the right
    ax.invert_xaxis()
    
    plt.tight_layout()
    
    return fig, ax


def create_performance_stability_comparison(df):
    """
    Create dual-axis plot comparing performance rank vs stability rank.
    
    Shows whether top performers are also most stable.
    """
    print("\n" + "="*70)
    print("CREATING PERFORMANCE-STABILITY COMPARISON")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by performance rank
    df_sorted = df.sort_values('performance_rank')
    
    x = range(len(df_sorted))
    width = 0.35
    
    # Performance rank bars
    bars1 = ax.bar(
        [i - width/2 for i in x],
        df_sorted['performance_rank'],
        width,
        label='Performance Rank',
        color='#2E86AB',
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    
    # Stability rank bars
    bars2 = ax.bar(
        [i + width/2 for i in x],
        df_sorted['stability_rank'],
        width,
        label='Stability Rank',
        color='#F18F01',
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    
    # X-axis: model names
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right', fontsize=10)
    
    # Y-axis
    ax.set_ylabel('Rank (1 = Best)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model (sorted by performance)', fontsize=12, fontweight='bold')
    
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    
    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, axis='y', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add note
    ax.text(
        0.02, 0.98,
        'Lower rank = Better',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    return fig, ax


def main():
    """Generate all stability visualizations."""
    print("="*70)
    print("MODEL STABILITY VISUALIZATION")
    print("="*70)
    
    # Load data
    df, correlation = load_stability_data()
    
    # 1. Main scatter plot: Mean Performance vs Stability
    fig1, ax1 = create_stability_scatter_plot(df)
    save_figure(fig1, "model_stability_scatter")
    
    # 2. Stability bar chart
    fig2, ax2 = create_stability_bar_chart(df)
    save_figure(fig2, "model_stability_ranking")
    
    # 3. Performance-Stability comparison
    fig3, ax3 = create_performance_stability_comparison(df)
    save_figure(fig3, "performance_stability_comparison")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated 3 publication-quality figures:")
    print("  1. Stability scatter plot (mean F1 vs CV)")
    print("  2. Stability ranking bar chart")
    print("  3. Performance-stability comparison")
    print("\nAll figures saved in PNG, PDF, and EPS formats")
    print(f"\nKey insight: Correlation between performance and stability: {correlation:.4f}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
