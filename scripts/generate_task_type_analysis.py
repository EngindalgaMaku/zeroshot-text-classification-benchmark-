"""
Generate Task-Type Analysis Figure
Publication-quality grouped bar chart comparing models across task types.

Outputs:
  results/plots/task_type_analysis.pdf
  results/plots/task_type_analysis.eps
  results/plots/task_type_analysis.png
  reports/TASK_TYPE_ANALYSIS.pdf
"""

import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("results/MULTI_DATASET_RESULTS.csv")
PLOT_DIR = Path("results/plots")
REPORTS_DIR = Path("reports")

# Dataset → task type mapping
TASK_TYPE_MAP = {
    "AG News":          "Topic",
    "20 Newsgroups":    "Topic",
    "Yahoo Answers":    "Topic",
    "DBpedia-14":       "Entity",
    "Banking77":        "Intent",
    "Twitter Financial": "Sentiment",
    "GoEmotions":       "Emotion",
    "IMDB":             "Sentiment",
    "SST-2":            "Sentiment",
}

TASK_ORDER = ["Topic", "Entity", "Intent", "Sentiment", "Emotion"]

# Consistent model color palette
MODEL_COLORS = {
    "BGE-M3":     "#1f77b4",  # blue
    "E5-large":   "#ff7f0e",  # orange
    "INSTRUCTOR": "#2ca02c",  # green
    "Jina v5":    "#d62728",  # red
    "MPNet":      "#9467bd",  # purple
    "Nomic-MoE":  "#8c564b",  # brown
    "Qwen3":      "#e377c2",  # pink
}


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["macro_f1"] = df["macro_f1"].astype(float)
    df["task_type"] = df["dataset"].map(TASK_TYPE_MAP)
    unmapped = df[df["task_type"].isna()]["dataset"].unique()
    if len(unmapped):
        print(f"  Warning: unmapped datasets skipped: {list(unmapped)}")
    df = df.dropna(subset=["task_type"])
    return df


def aggregate_by_task_type(df: pd.DataFrame):
    """
    For each (model, task_type) pair:
      - average macro_f1 across datasets within that task type
      - compute std across datasets (used as error bar; NaN when only 1 dataset)

    Returns two DataFrames indexed by task_type, columns = models:
      means_pivot  – mean Macro-F1 (%)
      stds_pivot   – std across datasets (NaN if only 1 dataset)
    """
    # First average per (model, dataset) in case of duplicates
    per_dataset = (
        df.groupby(["model", "dataset", "task_type"])["macro_f1"]
        .mean()
        .reset_index()
    )

    # Then aggregate across datasets within each task type
    agg = (
        per_dataset.groupby(["model", "task_type"])["macro_f1"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # std is NaN when only 1 dataset — keep as-is (no error bar drawn)
    means_pivot = agg.pivot(index="task_type", columns="model", values="mean")
    stds_pivot  = agg.pivot(index="task_type", columns="model", values="std")

    return means_pivot, stds_pivot


# ---------------------------------------------------------------------------
# Figure drawing
# ---------------------------------------------------------------------------

def draw_grouped_bar_chart(
    means_pivot: pd.DataFrame,
    stds_pivot: pd.DataFrame,
    model_order: list[str],
) -> plt.Figure:
    """
    Grouped bar chart: x-axis = task types, within each group one bar per model.
    Error bars show std across datasets (only for task types with >1 dataset).
    """
    task_types = [t for t in TASK_ORDER if t in means_pivot.index]
    n_tasks  = len(task_types)
    n_models = len(model_order)

    bar_width = 0.10
    group_gap = 0.15          # extra space between groups
    group_width = n_models * bar_width + group_gap
    group_centers = np.arange(n_tasks) * group_width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(model_order):
        if model not in means_pivot.columns:
            continue

        offsets = group_centers + (i - n_models / 2 + 0.5) * bar_width
        heights = [means_pivot.loc[t, model] if t in means_pivot.index else 0.0
                   for t in task_types]
        errors  = [stds_pivot.loc[t, model]
                   if (t in stds_pivot.index and not np.isnan(stds_pivot.loc[t, model]))
                   else 0.0
                   for t in task_types]

        ax.bar(
            offsets,
            heights,
            width=bar_width,
            color=MODEL_COLORS.get(model, "#333333"),
            label=model,
            edgecolor="white",
            linewidth=0.4,
        )

        # Draw error bars only where std > 0
        for x, h, e in zip(offsets, heights, errors):
            if e > 0:
                ax.errorbar(
                    x, h, yerr=e,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                )

    # x-axis ticks at group centers
    ax.set_xticks(group_centers)
    ax.set_xticklabels(task_types, fontsize=11)

    ax.set_ylabel("Mean Macro-F1 (%)", fontsize=12)
    ax.set_xlabel("")          # task type labels are self-explanatory

    # y-axis range: start at 0, end slightly above max value
    all_vals = means_pivot.values[~np.isnan(means_pivot.values)]
    ymax = max(all_vals) + 8
    ax.set_ylim(0, ymax)

    # y-axis grid only
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside the plot (upper right)
    ax.legend(
        title="Model",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename_base: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = PLOT_DIR / f"{filename_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PNG : {png_path}")

    pdf_path = PLOT_DIR / f"{filename_base}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  Saved PDF : {pdf_path}")

    eps_path = PLOT_DIR / f"{filename_base}.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"  Saved EPS : {eps_path}")

    return pdf_path


def copy_to_reports(pdf_path: Path, dest_name: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REPORTS_DIR / dest_name
    shutil.copy2(pdf_path, dest)
    print(f"  Copied to : {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Task-Type Analysis — Grouped Bar Chart")
    print("=" * 60)

    print(f"\nLoading data from {DATA_PATH} …")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets")

    means_pivot, stds_pivot = aggregate_by_task_type(df)

    # Use model order consistent with MODEL_COLORS definition
    model_order = [m for m in MODEL_COLORS if m in means_pivot.columns]

    print("\nPer-model means by task type (%):")
    print(means_pivot[model_order].round(1).to_string())

    print("\nDrawing grouped bar chart …")
    fig = draw_grouped_bar_chart(means_pivot, stds_pivot, model_order)

    print("\nSaving figures …")
    pdf_path = save_figure(fig, "task_type_analysis")
    copy_to_reports(pdf_path, "TASK_TYPE_ANALYSIS.pdf")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
