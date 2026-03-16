"""
Task Characteristics Visualizations

Generates publication-quality scatter plots showing relationships between
task characteristics and model performance (Macro-F1):
  - num_classes vs Macro-F1
  - avg_text_length vs Macro-F1
  - label_similarity vs Macro-F1

Different colors are used for different models.

**Validates: Requirements 8.4**
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent color palette for models (colorblind-friendly)
MODEL_COLORS = {
    "BGE-M3":     "#1f77b4",  # blue
    "E5-large":   "#ff7f0e",  # orange
    "INSTRUCTOR": "#2ca02c",  # green
    "Jina v5":    "#d62728",  # red
    "MPNet":      "#9467bd",  # purple
    "Nomic-MoE":  "#8c564b",  # brown
    "Qwen3":      "#e377c2",  # pink
}

MODEL_MARKERS = {
    "BGE-M3":     "o",
    "E5-large":   "s",
    "INSTRUCTOR": "^",
    "Jina v5":    "D",
    "MPNet":      "v",
    "Nomic-MoE":  "P",
    "Qwen3":      "*",
}


def load_data():
    """Load merged characteristics + results data."""
    chars_path = Path("results/task_characteristics/task_characteristics.csv")
    results_path = Path("results/MULTI_DATASET_RESULTS.csv")

    if not chars_path.exists():
        raise FileNotFoundError(
            "Task characteristics not found. Run compute_task_characteristics.py first."
        )

    chars_df = pd.read_csv(chars_path)
    results_df = pd.read_csv(results_path)

    # Drop num_classes from results if present (use chars_df version)
    results_clean = results_df.drop(columns=["num_classes"], errors="ignore")
    merged = results_clean.merge(chars_df, on="dataset", how="inner")

    print(f"Loaded {len(merged)} data points "
          f"({merged['dataset'].nunique()} datasets × {merged['model'].nunique()} models)")
    return merged


def add_regression_line(ax, x, y, color="gray", linestyle="--", alpha=0.6):
    """Add a linear regression line to the axes."""
    if len(x) < 2:
        return
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linestyle=linestyle,
            linewidth=1.5, alpha=alpha, zorder=1)
    return r, p


def scatter_characteristic(df, char_col, x_label, title, filename_base, log_x=False):
    """
    Create a scatter plot of one task characteristic vs Macro-F1.

    Each model gets a distinct color and marker. A global regression line
    is drawn over all points.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    models = sorted(df["model"].unique())

    for model in models:
        mdf = df[df["model"] == model]
        x = np.log10(mdf[char_col]) if log_x else mdf[char_col]
        ax.scatter(
            x,
            mdf["macro_f1"],
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
            label=model,
            zorder=3,
        )

    # Global regression line
    x_all = np.log10(df[char_col]) if log_x else df[char_col]
    y_all = df["macro_f1"]
    result = add_regression_line(ax, x_all.values, y_all.values)
    if result:
        r, p = result
        sig = "p<0.05" if p < 0.05 else f"p={p:.3f}"
        ax.text(
            0.97, 0.05,
            f"r = {r:+.3f} ({sig})",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

    # Axis labels
    x_label_full = f"log₁₀({x_label})" if log_x else x_label
    ax.set_xlabel(x_label_full, fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")

    # Grid
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        ncol=2,
    )
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    return fig


def create_combined_figure(df):
    """
    Create a 1×3 panel figure with all three scatter plots side by side.
    This is the main publication figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    configs = [
        ("num_classes",      "Number of Classes",         False),
        ("avg_text_length",  "Average Text Length (chars)", True),
        ("label_similarity", "Label Semantic Similarity",  False),
    ]

    models = sorted(df["model"].unique())

    for ax, (char_col, x_label, log_x) in zip(axes, configs):
        for model in models:
            mdf = df[df["model"] == model]
            x = np.log10(mdf[char_col]) if log_x else mdf[char_col]
            ax.scatter(
                x,
                mdf["macro_f1"],
                color=MODEL_COLORS.get(model, "gray"),
                marker=MODEL_MARKERS.get(model, "o"),
                s=60,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.4,
                label=model,
                zorder=3,
            )

        # Regression line
        x_all = np.log10(df[char_col]) if log_x else df[char_col]
        y_all = df["macro_f1"]
        result = add_regression_line(ax, x_all.values, y_all.values)
        if result:
            r, p = result
            sig = "p<0.05" if p < 0.05 else f"p={p:.3f}"
            ax.text(
                0.97, 0.05,
                f"r = {r:+.3f} ({sig})",
                transform=ax.transAxes,
                fontsize=9,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8),
            )

        x_label_full = f"log₁₀({x_label})" if log_x else x_label
        ax.set_xlabel(x_label_full, fontsize=11, fontweight="bold")
        ax.set_ylabel("Macro-F1 Score", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # Shared legend below the figure
    handles = [
        mpatches.Patch(
            color=MODEL_COLORS.get(m, "gray"),
            label=m,
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(models),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    return fig


def save_figure(fig, filename_base):
    """Save figure in PNG, PDF, and EPS formats."""
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ PNG: {png_path}")

    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  ✅ PDF: {pdf_path}")

    eps_path = output_dir / f"{filename_base}.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"  ✅ EPS: {eps_path}")


def main():
    """Generate all task characteristics visualizations."""
    print("=" * 70)
    print("TASK CHARACTERISTICS VISUALIZATIONS")
    print("=" * 70)

    df = load_data()

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. num_classes vs Macro-F1
    print("\n[1/4] num_classes vs Macro-F1")
    fig1 = scatter_characteristic(
        df, "num_classes", "Number of Classes",
        "Number of Classes vs Macro-F1",
        "task_char_num_classes",
        log_x=False,
    )
    save_figure(fig1, "task_char_num_classes")
    plt.close(fig1)

    # 2. avg_text_length vs Macro-F1
    print("\n[2/4] avg_text_length vs Macro-F1")
    fig2 = scatter_characteristic(
        df, "avg_text_length", "Average Text Length (chars)",
        "Average Text Length vs Macro-F1",
        "task_char_text_length",
        log_x=True,
    )
    save_figure(fig2, "task_char_text_length")
    plt.close(fig2)

    # 3. label_similarity vs Macro-F1
    print("\n[3/4] label_similarity vs Macro-F1")
    fig3 = scatter_characteristic(
        df, "label_similarity", "Label Semantic Similarity",
        "Label Semantic Similarity vs Macro-F1",
        "task_char_label_similarity",
        log_x=False,
    )
    save_figure(fig3, "task_char_label_similarity")
    plt.close(fig3)

    # 4. Combined 1×3 panel figure
    print("\n[4/4] Combined panel figure")
    fig4 = create_combined_figure(df)
    save_figure(fig4, "task_characteristics_combined")
    plt.close(fig4)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated 4 figures (each in PNG, PDF, EPS):")
    print("  1. task_char_num_classes      — num_classes vs Macro-F1")
    print("  2. task_char_text_length      — text_length vs Macro-F1 (log scale)")
    print("  3. task_char_label_similarity — label_similarity vs Macro-F1")
    print("  4. task_characteristics_combined — all three in one panel")


if __name__ == "__main__":
    main()
