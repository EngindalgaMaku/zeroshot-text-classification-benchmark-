"""Task × Label Interaction Analyzer.

Computes average Macro-F1 per (task_type, label_mode) combination, ΔF1 gains,
and produces heatmap and bar-plot visualisations.

Usage:
    python scripts/analyze_task_interaction.py [--results-dir DIR] [--output-dir DIR]

Output:
    reports/task_interaction/task_label_means.csv
    reports/task_interaction/delta_f1.csv
    reports/task_interaction/heatmap.png
    reports/task_interaction/delta_f1_gain.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset → task_type mapping (Requirements 5.5)
DATASET_TASK_TYPE: dict[str, str] = {
    "ag_news": "topic",
    "dbpedia_14": "topic",
    "yahoo_answers_topics": "topic",
    "SetFit/20_newsgroups": "topic",
    "banking77": "intent",
    "go_emotions": "emotion",
    "imdb": "sentiment",
    "sst2": "sentiment",
    "zeroshot/twitter-financial-news-sentiment": "sentiment",
}

# label_mode string → code
# Supports both legacy keys (name_only/description/multi_description) and
# llm_descriptions experiment keys (l1/l2/l3)
LABEL_MODE_CODE: dict[str, str] = {
    "name_only": "L1",
    "l1": "L1",
    "description": "L2",
    "l2": "L2",
    "multi_description": "L3",
    "l3": "L3",
}


def load_macro_f1_from_results(results_dir: Path) -> pd.DataFrame:
    """Read Macro-F1 values from all *_metrics.json files under results_dir.

    Returns a DataFrame with columns: dataset, label_mode, model, macro_f1.
    """
    records = []
    json_files = sorted(results_dir.glob("**/*_metrics.json"))

    if not json_files:
        logger.warning("No *_metrics.json files found in '%s'.", results_dir)
        return pd.DataFrame(columns=["dataset", "label_mode", "model", "macro_f1"])

    for fpath in json_files:
        logger.info("Reading result file: %s", fpath)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read '%s': %s", fpath, e)
            continue

        dataset = data.get("dataset")
        label_mode = data.get("label_mode")
        macro_f1 = data.get("macro_f1")
        model = data.get("biencoder") or data.get("reranker") or "unknown"

        if dataset is None or label_mode is None or macro_f1 is None:
            logger.warning("Missing fields in '%s' — skipping.", fpath)
            continue

        records.append(
            {
                "dataset": dataset,
                "label_mode": label_mode,
                "model": model,
                "macro_f1": float(macro_f1),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Loaded %d result records from %d files.", len(df), len(json_files))
    return df


class TaskInteractionAnalyzer:
    """Analyses task_type × label_mode interactions from Macro-F1 results.

    Args:
        df: DataFrame with columns: dataset, label_mode, model, macro_f1
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Map label_mode strings to L1/L2/L3 codes
        df = df.copy()
        df["label_mode_code"] = df["label_mode"].map(LABEL_MODE_CODE)
        df = df.dropna(subset=["label_mode_code"])

        # Map datasets to task_types; drop unknown datasets with a warning
        df["task_type"] = df["dataset"].map(DATASET_TASK_TYPE)
        unknown = df[df["task_type"].isna()]["dataset"].unique()
        for ds in unknown:
            logger.warning("Dataset '%s' has no task_type mapping — skipping.", ds)
        df = df.dropna(subset=["task_type"])

        self.df = df

    def compute_task_label_means(self) -> pd.DataFrame:
        """Compute average Macro-F1 per (task_type, label_mode) combination.

        Returns a DataFrame with columns: task_type, label_mode, avg_macro_f1.
        Missing combinations are represented as NaN rows.
        Requirements: 5.1, 5.6
        """
        task_types = sorted(set(DATASET_TASK_TYPE.values()))
        label_modes = ["L1", "L2", "L3"]

        avg = (
            self.df.groupby(["task_type", "label_mode_code"])["macro_f1"]
            .mean()
            .reset_index()
            .rename(columns={"label_mode_code": "label_mode", "macro_f1": "avg_macro_f1"})
        )

        # Build full grid so missing combos appear as NaN
        full_index = pd.MultiIndex.from_product(
            [task_types, label_modes], names=["task_type", "label_mode"]
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        result = full_df.merge(avg, on=["task_type", "label_mode"], how="left")
        return result

    def compute_delta_f1(self) -> pd.DataFrame:
        """Compute ΔF1(L2-L1) and ΔF1(L3-L1) per dataset.

        Returns a DataFrame with columns:
            dataset, task_type, delta_L2_L1, delta_L3_L1
        NaN when L1 baseline is missing for a dataset.
        Requirements: 5.3
        """
        # Average across models per (dataset, label_mode_code)
        per_dataset = (
            self.df.groupby(["dataset", "task_type", "label_mode_code"])["macro_f1"]
            .mean()
            .reset_index()
        )

        # Pivot to wide format
        pivot = per_dataset.pivot_table(
            index=["dataset", "task_type"],
            columns="label_mode_code",
            values="macro_f1",
        ).reset_index()
        pivot.columns.name = None

        # Ensure L1/L2/L3 columns exist even if data is absent
        for col in ["L1", "L2", "L3"]:
            if col not in pivot.columns:
                pivot[col] = np.nan

        pivot["delta_L2_L1"] = pivot["L2"] - pivot["L1"]
        pivot["delta_L3_L1"] = pivot["L3"] - pivot["L1"]

        result = pivot[["dataset", "task_type", "delta_L2_L1", "delta_L3_L1"]].copy()
        return result


def plot_heatmap(means_df: pd.DataFrame, output_path: Path) -> None:
    """Produce task_type × label_mode heatmap of average Macro-F1.

    Requirements: 5.2, 7.5
    """
    pivot = means_df.pivot(index="task_type", columns="label_mode", values="avg_macro_f1")
    # Ensure column order L1 → L2 → L3
    cols = [c for c in ["L1", "L2", "L3"] if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Avg Macro-F1"},
    )
    ax.set_title("Average Macro-F1 by Task Type × Label Mode")
    ax.set_xlabel("Label Mode")
    ax.set_ylabel("Task Type")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)


def plot_delta_f1_gain(delta_df: pd.DataFrame, output_path: Path) -> None:
    """Produce grouped bar chart of ΔF1 gains per dataset.

    Only includes datasets where L1 baseline exists (non-NaN deltas).
    Requirements: 5.4, 7.5
    """
    # Keep only rows where at least one delta is available
    plot_df = delta_df.dropna(subset=["delta_L2_L1", "delta_L3_L1"], how="all").copy()

    if plot_df.empty:
        logger.warning("No ΔF1 data available — skipping delta_f1_gain plot.")
        return

    # Shorten dataset names for readability
    def shorten(name: str) -> str:
        parts = name.split("/")
        return parts[-1] if len(parts) > 1 else name

    plot_df["short_name"] = plot_df["dataset"].apply(shorten)

    datasets = plot_df["short_name"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.2), 5))

    bars_l2 = ax.bar(
        x - width / 2,
        plot_df["delta_L2_L1"].fillna(0),
        width,
        label="ΔF1 (L2−L1)",
        color="steelblue",
        alpha=0.85,
    )
    bars_l3 = ax.bar(
        x + width / 2,
        plot_df["delta_L3_L1"].fillna(0),
        width,
        label="ΔF1 (L3−L1)",
        color="darkorange",
        alpha=0.85,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("ΔF1")
    ax.set_title("ΔF1 Gain over L1 Baseline per Dataset")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ΔF1 gain bar plot → %s", output_path)


def run_analysis(
    results_dir: str = "results",
    output_dir: str = "reports/task_interaction",
) -> None:
    """Run the full task × label interaction analysis pipeline.

    Args:
        results_dir: Directory containing *_metrics.json result files.
        output_dir:  Directory to write output CSVs and PNGs.

    Requirements: 7.2, 7.3, 7.6
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df = load_macro_f1_from_results(results_path)

    if df.empty:
        logger.error(
            "No result data loaded from '%s'. "
            "Ensure *_metrics.json files are present.",
            results_path,
        )
        return

    analyzer = TaskInteractionAnalyzer(df)

    # --- Task-label means ---
    means_df = analyzer.compute_task_label_means()

    # Pivot for CSV: rows=task_type, cols=L1/L2/L3
    means_pivot = means_df.pivot(
        index="task_type", columns="label_mode", values="avg_macro_f1"
    ).reset_index()
    means_pivot.columns.name = None
    for col in ["L1", "L2", "L3"]:
        if col not in means_pivot.columns:
            means_pivot[col] = np.nan

    means_csv = output_path / "task_label_means.csv"
    means_pivot.to_csv(means_csv, index=False)
    logger.info("Saved task_label_means → %s", means_csv)

    # --- ΔF1 ---
    delta_df = analyzer.compute_delta_f1()

    delta_csv = output_path / "delta_f1.csv"
    delta_df.to_csv(delta_csv, index=False)
    logger.info("Saved delta_f1 → %s", delta_csv)

    # --- Visualisations ---
    heatmap_png = output_path / "heatmap.png"
    plot_heatmap(means_df, heatmap_png)

    delta_png = output_path / "delta_f1_gain.png"
    plot_delta_f1_gain(delta_df, delta_png)

    # --- Summary ---
    print("\nTask × Label Means (avg Macro-F1):")
    print(means_pivot.to_string(index=False))
    print("\nΔF1 per Dataset:")
    print(delta_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse task × label mode interactions from Macro-F1 results."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing *_metrics.json result files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/task_interaction",
        help="Output directory for CSVs and PNGs (default: reports/task_interaction)",
    )
    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
