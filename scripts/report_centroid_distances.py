"""Centroid Distance Analysis Report and Visualization.

Loads the centroid distance metrics CSV produced by compute_centroid_distances.py
and generates:

  1. Summary table (CSV + LaTeX) comparing intra-label variance and inter-label
     separation across L1/L2/L3 levels, with a label_collapse flag.

  2. Line plots showing how inter-label separation changes from L1 → L3 per
     model (one subplot per model, one line per dataset).  Models where L3
     causes label collapse are highlighted with a red background tint.

  3. Label-collapse detection figure: a heatmap/bar chart showing which
     model × dataset combinations exhibit label collapse (L3 inter_sep < L1).

  4. Text summary printed to stdout listing models with collapse and the
     average inter-separation change (L1 -> L3) per model.

Usage:
    python scripts/report_centroid_distances.py \\
        [--input results/centroid_distances.csv] \\
        [--output-dir results/plots] \\
        [--tables-dir results/tables]

Requirements: 7.3, 7.4, 7.5
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------
PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(PUBLICATION_STYLE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "bge":        "BGE-M3",
    "e5":         "E5-large",
    "instructor": "INSTRUCTOR",
    "jina_v5":    "Jina v5",
    "mpnet":      "MPNet",
    "nomic":      "Nomic-MoE",
    "qwen3":      "Qwen3",
}

# Ordered list of model keys (for consistent subplot layout)
MODEL_ORDER = ["bge", "e5", "instructor", "jina_v5", "mpnet", "nomic", "qwen3"]

LEVEL_ORDER = ["L1", "L2", "L3"]

# Colour palette for datasets (up to 12 distinct colours)
DATASET_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

COLLAPSE_RED   = "#ffcccc"   # background tint for collapsed subplots
COLLAPSE_COLOR = "#d62728"   # red for collapse cells
IMPROVE_COLOR  = "#2ca02c"   # green for improvement cells
NEUTRAL_COLOR  = "#aaaaaa"   # gray for neutral cells


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(input_path: Path) -> pd.DataFrame:
    """Load centroid distances CSV with a helpful error if missing."""
    if not input_path.exists():
        print(
            f"\nERROR: Input file not found: {input_path}\n"
            "Please generate it first by running:\n"
            "    python scripts/compute_centroid_distances.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(input_path)
    required = {"model", "dataset", "label_level", "intra_variance", "inter_separation"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"\nERROR: CSV is missing required columns: {missing}\n"
            "Please re-run compute_centroid_distances.py to regenerate the file.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Loaded %d rows from %s", len(df), input_path)
    return df


def save_figure(fig: plt.Figure, output_dir: Path, filename_base: str) -> None:
    """Save figure in PDF, EPS, and PNG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        path = output_dir / f"{filename_base}.{ext}"
        kwargs: dict = {"bbox_inches": "tight"}
        if ext != "png":
            kwargs["format"] = ext
        fig.savefig(path, dpi=300, **kwargs)
        logger.info("  Saved %s: %s", ext.upper(), path)


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-(model, dataset) summary with inter_sep at each level and collapse flag.

    Columns: model, dataset, intra_variance_mean,
             inter_sep_L1, inter_sep_L2, inter_sep_L3, label_collapse
    """
    rows = []
    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        level_sep = grp.set_index("label_level")["inter_separation"].to_dict()
        intra_mean = grp["intra_variance"].mean()

        inter_l1 = level_sep.get("L1", np.nan)
        inter_l2 = level_sep.get("L2", np.nan)
        inter_l3 = level_sep.get("L3", np.nan)

        # Label collapse: L3 inter-separation is lower than L1
        collapse = bool(
            not np.isnan(inter_l1) and not np.isnan(inter_l3) and inter_l3 < inter_l1
        )

        rows.append({
            "model":              model,
            "dataset":            dataset,
            "intra_variance_mean": round(intra_mean, 6) if not np.isnan(intra_mean) else np.nan,
            "inter_sep_L1":       round(inter_l1, 6) if not np.isnan(inter_l1) else np.nan,
            "inter_sep_L2":       round(inter_l2, 6) if not np.isnan(inter_l2) else np.nan,
            "inter_sep_L3":       round(inter_l3, 6) if not np.isnan(inter_l3) else np.nan,
            "label_collapse":     collapse,
        })

    summary = pd.DataFrame(rows)

    # Sort by model order then dataset
    model_rank = {k: i for i, k in enumerate(MODEL_ORDER)}
    summary["_model_rank"] = summary["model"].map(lambda m: model_rank.get(m, 99))
    summary = summary.sort_values(["_model_rank", "dataset"]).drop(columns=["_model_rank"])
    summary = summary.reset_index(drop=True)
    return summary


def save_summary_table(summary: pd.DataFrame, tables_dir: Path) -> None:
    """Save summary table as CSV and LaTeX."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "centroid_distance_summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("Saved summary CSV → %s", csv_path)

    # LaTeX version — use display names for models
    latex_df = summary.copy()
    latex_df["model"] = latex_df["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    latex_df = latex_df.rename(columns={
        "model":               "Model",
        "dataset":             "Dataset",
        "intra_variance_mean": "Intra Var.",
        "inter_sep_L1":        "Inter Sep. L1",
        "inter_sep_L2":        "Inter Sep. L2",
        "inter_sep_L3":        "Inter Sep. L3",
        "label_collapse":      "Collapse",
    })

    tex_path = tables_dir / "centroid_distance_summary.tex"
    latex_str = latex_df.to_latex(
        index=False,
        float_format="%.4f",
        caption=(
            "Summary of centroid distance metrics per model and dataset. "
            "Intra Var.\\ is the mean cosine distance between L1/L2/L3 centroids "
            "of the same class. Inter Sep.\\ is the mean pairwise cosine distance "
            "between different class centroids at each label level. "
            "Collapse indicates whether L3 inter-separation is lower than L1."
        ),
        label="tab:centroid_distance_summary",
        na_rep="--",
    )
    tex_path.write_text(latex_str, encoding="utf-8")
    logger.info("Saved summary LaTeX → %s", tex_path)


# ---------------------------------------------------------------------------
# 2. Inter-separation line plots (one subplot per model)
# ---------------------------------------------------------------------------

def plot_inter_separation(df: pd.DataFrame, output_dir: Path, summary: pd.DataFrame) -> None:
    """Generate line plots of inter-label separation across L1/L2/L3 per model."""
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    datasets = sorted(df["dataset"].unique())
    dataset_colors = {ds: DATASET_PALETTE[i % len(DATASET_PALETTE)] for i, ds in enumerate(datasets)}

    n_models = len(models_present)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0), squeeze=False)

    # Build collapse lookup: {(model, dataset): bool}
    collapse_lookup = {
        (row["model"], row["dataset"]): row["label_collapse"]
        for _, row in summary.iterrows()
    }

    # Determine if a model has ANY collapse across its datasets
    model_has_collapse = {}
    for model in models_present:
        model_has_collapse[model] = any(
            collapse_lookup.get((model, ds), False) for ds in datasets
        )

    for idx, model in enumerate(models_present):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx][col_idx]

        model_df = df[df["model"] == model]
        display_name = MODEL_DISPLAY.get(model, model)

        # Red background tint if any dataset collapses for this model
        if model_has_collapse[model]:
            ax.set_facecolor(COLLAPSE_RED)

        for ds in datasets:
            ds_df = model_df[model_df["dataset"] == ds].set_index("label_level")
            # Only plot if we have at least 2 levels
            available = [lv for lv in LEVEL_ORDER if lv in ds_df.index]
            if len(available) < 2:
                continue

            y_vals = [ds_df.loc[lv, "inter_separation"] for lv in available]
            collapsed = collapse_lookup.get((model, ds), False)
            lw = 2.0 if collapsed else 1.2
            ls = "-" if not collapsed else "--"
            ax.plot(
                available, y_vals,
                color=dataset_colors[ds],
                linewidth=lw,
                linestyle=ls,
                marker="o",
                markersize=4,
                label=ds,
            )

        title = display_name
        if model_has_collapse[model]:
            title += " [!]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Label Level", fontsize=9)
        ax.set_ylabel("Inter-Sep.", fontsize=9)
        ax.set_xticks(LEVEL_ORDER)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].set_visible(False)

    # Shared legend below the figure
    handles = [
        plt.Line2D([0], [0], color=dataset_colors[ds], linewidth=1.5, label=ds)
        for ds in datasets
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(datasets), 5),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Collapse annotation note
    if any(model_has_collapse.values()):
        fig.text(
            0.5, -0.06,
            "[!] = model has at least one dataset with label collapse (L3 inter-sep < L1)."
            "  Red background = collapse detected.  Dashed lines = collapsing datasets.",
            ha="center", fontsize=8, style="italic",
        )

    fig.suptitle(
        "Inter-Label Separation Across Label Levels (L1 → L2 → L3) per Model",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "centroid_inter_separation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Label collapse detection figure
# ---------------------------------------------------------------------------

def plot_label_collapse(summary: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap showing which model × dataset combinations exhibit label collapse."""
    models_present = [m for m in MODEL_ORDER if m in summary["model"].unique()]
    datasets = sorted(summary["dataset"].unique())

    # Build numeric matrix: -1 = collapse, 0 = neutral, +1 = improvement
    matrix = np.zeros((len(models_present), len(datasets)), dtype=float)
    for i, model in enumerate(models_present):
        for j, ds in enumerate(datasets):
            row = summary[(summary["model"] == model) & (summary["dataset"] == ds)]
            if row.empty:
                matrix[i, j] = np.nan
                continue
            l1 = row["inter_sep_L1"].values[0]
            l3 = row["inter_sep_L3"].values[0]
            if np.isnan(l1) or np.isnan(l3):
                matrix[i, j] = np.nan
            elif l3 < l1:
                matrix[i, j] = -1.0   # collapse
            elif l3 > l1:
                matrix[i, j] = 1.0    # improvement
            else:
                matrix[i, j] = 0.0    # neutral

    # Also compute the delta (L3 - L1) for annotation
    delta_matrix = np.full_like(matrix, np.nan)
    for i, model in enumerate(models_present):
        for j, ds in enumerate(datasets):
            row = summary[(summary["model"] == model) & (summary["dataset"] == ds)]
            if row.empty:
                continue
            l1 = row["inter_sep_L1"].values[0]
            l3 = row["inter_sep_L3"].values[0]
            if not (np.isnan(l1) or np.isnan(l3)):
                delta_matrix[i, j] = l3 - l1

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.1), max(3, len(models_present) * 0.7)))

    # Custom colormap: red → gray → green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "collapse_cmap",
        [COLLAPSE_COLOR, NEUTRAL_COLOR, IMPROVE_COLOR],
        N=256,
    )

    # Mask NaN cells
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")

    # Annotate cells with delta values
    for i in range(len(models_present)):
        for j in range(len(datasets)):
            val = delta_matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:+.3f}"
                color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels(
        [MODEL_DISPLAY.get(m, m) for m in models_present], fontsize=9
    )
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title(
        "Label Collapse Detection: Inter-Separation Change (L3 − L1)\n"
        "Red = collapse (L3 < L1)  |  Green = improvement  |  Gray = neutral",
        fontsize=10,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Direction", fontsize=8)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Collapse", "Neutral", "Improve"], fontsize=7)

    plt.tight_layout()
    save_figure(fig, output_dir, "centroid_label_collapse")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Text summary
# ---------------------------------------------------------------------------

def print_text_summary(summary: pd.DataFrame) -> None:
    """Print a text summary of label collapse and average inter-separation change."""
    print("\n" + "=" * 70)
    print("CENTROID DISTANCE ANALYSIS SUMMARY")
    print("=" * 70)

    # Models with label collapse on any dataset
    collapse_rows = summary[summary["label_collapse"] == True]
    if collapse_rows.empty:
        print("\nNo label collapse detected across any model × dataset combination.")
    else:
        print("\nModels with label collapse (L3 inter-sep < L1) on at least one dataset:")
        for model in MODEL_ORDER:
            model_collapse = collapse_rows[collapse_rows["model"] == model]
            if model_collapse.empty:
                continue
            display = MODEL_DISPLAY.get(model, model)
            datasets_collapsed = sorted(model_collapse["dataset"].tolist())
            print(f"  {display:15s}: {', '.join(datasets_collapsed)}")

    # Average inter-separation change (L1 -> L3) per model
    print("\nAverage inter-separation change (L3 - L1) per model:")
    for model in MODEL_ORDER:
        model_rows = summary[summary["model"] == model].copy()
        if model_rows.empty:
            continue
        valid = model_rows.dropna(subset=["inter_sep_L1", "inter_sep_L3"])
        if valid.empty:
            continue
        delta = (valid["inter_sep_L3"] - valid["inter_sep_L1"]).mean()
        direction = "improvement" if delta > 0 else "collapse"
        display = MODEL_DISPLAY.get(model, model)
        print(f"  {display:15s}: {delta:+.4f}  ({direction})")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate centroid distance analysis report and visualizations "
            "from the output of compute_centroid_distances.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="results/centroid_distances.csv",
        help="Path to centroid_distances.csv produced by compute_centroid_distances.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables",
        help="Directory for output tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)
    tables_dir  = Path(args.tables_dir)

    # Load data
    df = load_data(input_path)

    # 1. Build and save summary table
    logger.info("Building summary table…")
    summary = build_summary_table(df)
    save_summary_table(summary, tables_dir)

    # 2. Inter-separation line plots
    logger.info("Generating inter-separation line plots…")
    plot_inter_separation(df, output_dir, summary)

    # 3. Label collapse detection figure
    logger.info("Generating label collapse detection figure…")
    plot_label_collapse(summary, output_dir)

    # 4. Text summary
    print_text_summary(summary)

    logger.info("All outputs written.")
    logger.info("  Tables : %s", tables_dir)
    logger.info("  Figures: %s", output_dir)


if __name__ == "__main__":
    main()
