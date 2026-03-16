"""
Generate Label Formulation Comparison Figure
Compares name_only vs description label modes across datasets.

If name_only results are not yet available, generates a placeholder figure
showing description results with a note to run label formulation experiments.

Outputs:
  results/plots/label_formulation_comparison.pdf
  results/plots/label_formulation_comparison.eps
  results/plots/label_formulation_comparison.png
"""

import json
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
RAW_DIR   = Path("results/raw")
PLOT_DIR  = Path("results/plots")

# The 3 datasets selected for label formulation analysis (task 6.1)
TARGET_DATASETS = ["AG News", "Banking77", "GoEmotions"]

# Mapping from dataset display name → file key used in raw result filenames
DATASET_KEYS = {
    "AG News":    "ag_news",
    "Banking77":  "banking77",
    "GoEmotions": "go_emotions",
}

# Mapping from model display name → file key used in raw result filenames
MODEL_KEYS = {
    "BGE-M3":     "bge",
    "E5-large":   "e5",
    "INSTRUCTOR": "instructor",
    "Jina v5":    "jina_v5",
    "MPNet":      "mpnet",
    "Nomic-MoE":  "nomic",
    "Qwen3":      "qwen3",
}

# Colors for the two label modes (full comparison figure)
COLOR_DESCRIPTION = "#2E86AB"  # blue
COLOR_NAME_ONLY   = "#F18F01"  # orange

# Model color palette (placeholder figure)
MODEL_COLORS = {
    "BGE-M3":     "#1f77b4",
    "E5-large":   "#ff7f0e",
    "INSTRUCTOR": "#2ca02c",
    "Jina v5":    "#d62728",
    "MPNet":      "#9467bd",
    "Nomic-MoE":  "#8c564b",
    "Qwen3":      "#e377c2",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_description_data() -> pd.DataFrame:
    """Load description-mode results from the consolidated CSV."""
    df = pd.read_csv(DATA_PATH)
    df["macro_f1"] = df["macro_f1"].astype(float)
    return df[df["dataset"].isin(TARGET_DATASETS)].copy()


def try_load_name_only_results() -> dict[str, dict[str, float]] | None:
    """
    Attempt to load name_only metric files from results/raw/.

    Expected filename pattern: {dataset_key}_{model_key}_name_only_metrics.json
    Returns a nested dict {dataset_display_name: {model_display_name: macro_f1}}
    or None if no files are found.
    """
    results: dict[str, dict[str, float]] = {ds: {} for ds in TARGET_DATASETS}
    found_any = False

    for ds_name, ds_key in DATASET_KEYS.items():
        for model_name, model_key in MODEL_KEYS.items():
            fname = RAW_DIR / f"{ds_key}_{model_key}_name_only_metrics.json"
            if fname.exists():
                with open(fname) as f:
                    data = json.load(f)
                macro_f1 = data.get("macro_f1", data.get("macro_f1_score"))
                if macro_f1 is not None:
                    results[ds_name][model_name] = float(macro_f1) * 100 \
                        if float(macro_f1) <= 1.0 else float(macro_f1)
                    found_any = True

    return results if found_any else None


# ---------------------------------------------------------------------------
# Full comparison figure (both modes available)
# ---------------------------------------------------------------------------

def build_comparison_data(
    desc_df: pd.DataFrame,
    name_only_results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    For each dataset, compute the mean Macro-F1 across all models for both
    label modes.  Returns a DataFrame with columns:
      dataset, description_mean, name_only_mean, delta
    """
    rows = []
    for ds in TARGET_DATASETS:
        desc_vals = desc_df[desc_df["dataset"] == ds]["macro_f1"].values
        desc_mean = float(np.mean(desc_vals)) if len(desc_vals) > 0 else np.nan

        no_vals = list(name_only_results.get(ds, {}).values())
        no_mean = float(np.mean(no_vals)) if len(no_vals) > 0 else np.nan

        rows.append({
            "dataset":          ds,
            "description_mean": desc_mean,
            "name_only_mean":   no_mean,
            "delta":            desc_mean - no_mean,
        })
    return pd.DataFrame(rows)


def draw_full_comparison(comp_df: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side bar chart: description vs name_only per dataset.
    Delta (description − name_only) is annotated above each pair.
    """
    datasets = comp_df["dataset"].tolist()
    n = len(datasets)

    bar_width = 0.35
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_desc = ax.bar(
        x - bar_width / 2,
        comp_df["description_mean"],
        width=bar_width,
        color=COLOR_DESCRIPTION,
        label="description",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_no = ax.bar(
        x + bar_width / 2,
        comp_df["name_only_mean"],
        width=bar_width,
        color=COLOR_NAME_ONLY,
        label="name_only",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate delta above each pair
    for i, row in comp_df.iterrows():
        pair_top = max(row["description_mean"], row["name_only_mean"])
        delta_str = f"Δ={row['delta']:+.1f}"
        ax.text(
            x[i], pair_top + 1.5,
            delta_str,
            ha="center", va="bottom",
            fontsize=9, color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Mean Macro-F1 (%)", fontsize=12)

    all_vals = pd.concat([comp_df["description_mean"], comp_df["name_only_mean"]])
    ax.set_ylim(0, all_vals.max() + 12)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Label mode",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Placeholder figure (name_only not yet available)
# ---------------------------------------------------------------------------

def draw_placeholder(desc_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart showing description results per model per dataset.
    Includes a prominent note that name_only results are pending.
    """
    models = [m for m in MODEL_COLORS if m in desc_df["model"].unique()]
    n_datasets = len(TARGET_DATASETS)
    n_models   = len(models)

    bar_width  = 0.10
    group_gap  = 0.15
    group_width = n_models * bar_width + group_gap
    group_centers = np.arange(n_datasets) * group_width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        offsets = group_centers + (i - n_models / 2 + 0.5) * bar_width
        heights = []
        for ds in TARGET_DATASETS:
            val = desc_df[(desc_df["dataset"] == ds) & (desc_df["model"] == model)]["macro_f1"]
            heights.append(float(val.iloc[0]) if len(val) > 0 else 0.0)

        ax.bar(
            offsets,
            heights,
            width=bar_width,
            color=MODEL_COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(TARGET_DATASETS, fontsize=11)
    ax.set_ylabel("Macro-F1 — description mode (%)", fontsize=12)

    all_vals = desc_df["macro_f1"].values
    ax.set_ylim(0, max(all_vals) + 18)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # Prominent pending note
    ax.text(
        0.5, 0.97,
        "Note: name_only results not yet available.\n"
        "Run label formulation experiments to generate comparison.",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10,
        color="#8B0000",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                  edgecolor="#e0a800", linewidth=1.0, alpha=0.9),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename_base: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for ext, fmt in [("png", "png"), ("pdf", "pdf"), ("eps", "eps")]:
        path = PLOT_DIR / f"{filename_base}.{ext}"
        kwargs = dict(bbox_inches="tight")
        if fmt != "png":
            kwargs["format"] = fmt
        fig.savefig(path, dpi=300, **kwargs)
        print(f"  Saved {ext.upper():3s}: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Label Formulation Comparison Figure")
    print("=" * 60)

    print(f"\nLoading description results from {DATA_PATH} …")
    desc_df = load_description_data()
    print(f"  {len(desc_df)} rows for target datasets "
          f"({', '.join(TARGET_DATASETS)})")

    print("\nChecking for name_only result files in results/raw/ …")
    name_only_results = try_load_name_only_results()

    if name_only_results is not None:
        print("  name_only results FOUND — generating full comparison figure.")
        comp_df = build_comparison_data(desc_df, name_only_results)
        print("\nComparison summary (mean Macro-F1 %):")
        print(comp_df.to_string(index=False))
        fig = draw_full_comparison(comp_df)
        mode = "full comparison"
    else:
        print("  name_only results NOT found — generating placeholder figure.")
        print("  To generate the full comparison, run label formulation experiments first:")
        print("    e.g. python main.py --config experiments/ag_news_bge_name_only.yaml")
        fig = draw_placeholder(desc_df)
        mode = "placeholder"

    print(f"\nSaving {mode} figure …")
    save_figure(fig, "label_formulation_comparison")
    plt.close(fig)

    print(f"\nDone. Mode: {mode}")
    if name_only_results is None:
        print("\n[!] name_only results pending — run label formulation experiments first")


if __name__ == "__main__":
    main()
