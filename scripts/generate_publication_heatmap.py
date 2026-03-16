"""
Generate a publication-quality model × dataset Macro-F1 heatmap.

Outputs:
  results/plots/heatmap_publication.pdf
  results/plots/heatmap_publication.eps
  results/plots/heatmap_publication.png
  reports/F1_HEATMAP_PUBLICATION.pdf
"""

import shutil
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress font-substitution warnings that are harmless in CI
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

DATA_PATH = Path("results/MULTI_DATASET_RESULTS.csv")
PLOT_DIR = Path("results/plots")
REPORTS_DIR = Path("reports")
FILENAME_BASE = "heatmap_publication"


# ---------------------------------------------------------------------------
# Data loading & pivot
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["macro_f1"] = df["macro_f1"].astype(float)
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return model × dataset pivot of Macro-F1 (%)."""
    pivot = df.pivot_table(
        index="model", columns="dataset", values="macro_f1", aggfunc="mean"
    )
    return pivot


def add_mean_row_col(pivot: pd.DataFrame) -> pd.DataFrame:
    """Append a 'Mean' column (row averages) and a 'Mean' row (column averages)."""
    pivot = pivot.copy()
    pivot["Mean"] = pivot.mean(axis=1)
    mean_row = pivot.mean(axis=0)
    mean_row.name = "Mean"
    pivot = pd.concat([pivot, mean_row.to_frame().T])
    return pivot


def sort_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows (models) by mean Macro-F1 descending, columns (datasets) by
    mean Macro-F1 descending.  The 'Mean' row/column are kept at the end.
    """
    # Separate the Mean row/column
    data_rows = pivot.drop(index="Mean")
    data_cols = [c for c in pivot.columns if c != "Mean"]

    # Sort models by their mean (last column before we appended Mean row)
    data_rows = data_rows.sort_values("Mean", ascending=False)

    # Sort datasets by column mean (Mean row, data columns only)
    col_means = pivot.loc["Mean", data_cols].sort_values(ascending=False)
    sorted_cols = list(col_means.index) + ["Mean"]

    # Reassemble
    sorted_rows = list(data_rows.index) + ["Mean"]
    return pivot.loc[sorted_rows, sorted_cols]


# ---------------------------------------------------------------------------
# Figure drawing
# ---------------------------------------------------------------------------

def draw_heatmap(pivot: pd.DataFrame) -> plt.Figure:
    """
    Draw the publication heatmap.

    - Data cells: YlOrRd colour map
    - Mean row/column: slightly different shade (same cmap, separate normalisation)
    - Missing values (NaN): grey
    - Best model per dataset (column max, excluding Mean row): bold annotation
    """
    # Separate data block from Mean row/column for independent colour scaling
    data_rows = [r for r in pivot.index if r != "Mean"]
    data_cols = [c for c in pivot.columns if c != "Mean"]

    data_block = pivot.loc[data_rows, data_cols]
    mean_col = pivot.loc[data_rows, "Mean"]
    mean_row = pivot.loc["Mean", :]

    # Identify best model per dataset (column max in data block)
    best_model_per_dataset = data_block.idxmax(axis=0)  # Series: dataset → model

    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- draw the full heatmap (all rows × all cols) ----
    cmap = matplotlib.colormaps.get_cmap("Blues")
    cmap.set_bad(color="#cccccc")  # grey for NaN

    # Adjust vmin to make low values more visible (e.g., GoEmotions)
    # Instead of using absolute minimum, use a fixed lower bound
    vmin = 10  # Fixed minimum for better color contrast
    vmax = pivot.loc[data_rows, data_cols].max().max()

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,          # we'll annotate manually for bold control
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": "Macro-F1 (%)"},
    )

    # ---- manual annotations ----
    n_rows, n_cols = pivot.shape
    for row_idx, row_name in enumerate(pivot.index):
        for col_idx, col_name in enumerate(pivot.columns):
            val = pivot.iloc[row_idx, col_idx]
            if pd.isna(val):
                text = "—"
                weight = "normal"
                color = "#555555"
            else:
                text = f"{val:.1f}"
                # Bold if this cell is the best model for this dataset
                is_best = (
                    col_name in best_model_per_dataset.index
                    and best_model_per_dataset[col_name] == row_name
                )
                weight = "bold" if is_best else "normal"
                # Dark text on light cells, light text on dark cells
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = "white" if norm_val > 0.65 else "black"

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight=weight,
                color=color,
            )

    # ---- axis labels ----
    ax.set_xlabel("Dataset", fontsize=12, labelpad=8)
    ax.set_ylabel("Embedding Model", fontsize=12, labelpad=8)

    # Rotate x-tick labels for readability
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=30, ha="right", fontsize=10
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Draw a separator line before the Mean row and Mean column
    mean_row_idx = list(pivot.index).index("Mean")
    mean_col_idx = list(pivot.columns).index("Mean")

    ax.axhline(mean_row_idx, color="black", linewidth=1.5)
    ax.axvline(mean_col_idx, color="black", linewidth=1.5)

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
    print("Publication Heatmap Generator")
    print("=" * 60)

    print(f"\nLoading data from {DATA_PATH} …")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} rows loaded, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets")

    pivot_raw = build_pivot(df)
    pivot_with_means = add_mean_row_col(pivot_raw)
    pivot_sorted = sort_pivot(pivot_with_means)

    print("\nModel × dataset matrix (sorted):")
    print(pivot_sorted.round(1).to_string())

    print("\nDrawing heatmap …")
    fig = draw_heatmap(pivot_sorted)

    print("\nSaving figures …")
    pdf_path = save_figure(fig, FILENAME_BASE)
    copy_to_reports(pdf_path, "F1_HEATMAP_PUBLICATION.pdf")

    plt.close(fig)
    print("\nDone.")
    print(
        "\nSuggested caption:\n"
        "Macro-F1 scores (%) for seven sentence-embedding models across nine "
        "text classification datasets. Each cell shows the Macro-F1 score; "
        "bold values indicate the best-performing model for that dataset. "
        "The rightmost column and bottom row report per-model and per-dataset "
        "averages, respectively. Missing entry (20 Newsgroups × Qwen3) is "
        "shown in grey. Rows and columns are sorted by mean Macro-F1 "
        "(descending)."
    )


if __name__ == "__main__":
    main()
