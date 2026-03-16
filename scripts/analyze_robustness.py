"""Robustness Analyzer.

Computes |ΔF1(A-B)| for each (dataset, model) pair by comparing Macro-F1
from description_set_a and description_set_b experiments.

Usage:
    python scripts/analyze_robustness.py [--results-dir DIR] [--output-dir DIR]

Output:
    reports/robustness/robustness_scores.csv
    reports/robustness/robustness_heatmap.png
    reports/robustness/robustness_summary.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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

# Robustness threshold: |ΔF1| < 0.02 → "robust to description phrasing"
ROBUSTNESS_THRESHOLD = 0.02

# Expected datasets and models (for logging missing pairs)
EXPECTED_DATASETS = [
    "ag_news",
    "dbpedia_14",
    "yahoo_answers_topics",
    "banking77",
    "zeroshot/twitter-financial-news-sentiment",
    "SetFit/20_newsgroups",
    "imdb",
    "sst2",
    "go_emotions",
]

EXPECTED_MODELS = [
    "INSTRUCTOR-large",
    "bge-m3",
    "all-mpnet-base-v2",
    "nomic-embed-text",
    "e5-large-v2",
    "jina-embeddings-v3",
    "qwen3-embedding",
]

# label_mode values for Set A and Set B
SET_A_MODE = "description_set_a"
SET_B_MODE = "description_set_b"


def load_macro_f1_from_results(results_dir: Path) -> pd.DataFrame:
    """Read Macro-F1 values from all *_metrics.json files under results_dir.

    Returns a DataFrame with columns: dataset, label_mode, model, macro_f1, source_file.
    Requirements: 7.2, 9.3
    """
    records = []
    json_files = sorted(results_dir.glob("**/*_metrics.json"))

    if not json_files:
        logger.warning("No *_metrics.json files found in '%s'.", results_dir)
        return pd.DataFrame(columns=["dataset", "label_mode", "model", "macro_f1", "source_file"])

    for fpath in json_files:
        logger.info("Reading result file: %s", fpath)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read '%s': %s", fpath, e)
            continue

        dataset = data.get("dataset")
        label_mode = data.get("label_mode")
        macro_f1 = data.get("macro_f1")
        model = data.get("biencoder") or data.get("reranker") or "unknown"

        if dataset is None or label_mode is None or macro_f1 is None:
            logger.warning("Missing fields in '%s' -- skipping.", fpath)
            continue

        records.append({
            "dataset": dataset,
            "label_mode": label_mode,
            "model": model,
            "macro_f1": float(macro_f1),
            "source_file": str(fpath),
        })

    df = pd.DataFrame(records)
    logger.info("Loaded %d result records from %d files.", len(df), len(json_files))
    return df


class RobustnessAnalyzer:
    """Computes |ΔF1(A-B)| for each (dataset, model) pair.

    Reads Set A (description_set_a) and Set B (description_set_b) Macro-F1
    values from results/, computes the absolute difference per pair, and
    reports the overall average against the 0.02 robustness threshold.

    Missing Set B entries are logged; analysis proceeds with available data.

    Args:
        df: DataFrame with columns: dataset, label_mode, model, macro_f1

    Requirements: 9.3, 9.4, 9.7
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._log_missing_set_b()

    def _log_missing_set_b(self) -> None:
        """Log (dataset, model) pairs that have Set A but are missing Set B.

        Requirements: 9.7
        """
        set_a = self.df[self.df["label_mode"] == SET_A_MODE][["dataset", "model"]].drop_duplicates()
        set_b = self.df[self.df["label_mode"] == SET_B_MODE][["dataset", "model"]].drop_duplicates()

        if set_a.empty:
            logger.warning(
                "No '%s' results found. Robustness analysis requires Set A data.", SET_A_MODE
            )
            return

        # Find pairs in Set A but not in Set B
        merged = set_a.merge(set_b, on=["dataset", "model"], how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"][["dataset", "model"]]

        if missing.empty:
            logger.info("All Set A (dataset, model) pairs have corresponding Set B results.")
        else:
            for _, row in missing.iterrows():
                logger.warning(
                    "Missing Set B result for dataset='%s', model='%s' -- "
                    "this pair will be excluded from robustness scores.",
                    row["dataset"],
                    row["model"],
                )

    def compute_delta_f1(self) -> pd.DataFrame:
        """Compute |ΔF1(A-B)| for each (dataset, model) pair.

        Steps:
        1. Extract Set A and Set B Macro-F1 values.
        2. Inner-join on (dataset, model) — only pairs with both sets are included.
        3. Compute absolute difference.

        Returns:
            DataFrame with columns: dataset, model, f1_set_a, f1_set_b, delta_f1_abs
            Sorted by delta_f1_abs descending.

        Requirements: 9.3, 9.4
        """
        set_a = (
            self.df[self.df["label_mode"] == SET_A_MODE]
            .groupby(["dataset", "model"])["macro_f1"]
            .mean()
            .reset_index()
            .rename(columns={"macro_f1": "f1_set_a"})
        )

        set_b = (
            self.df[self.df["label_mode"] == SET_B_MODE]
            .groupby(["dataset", "model"])["macro_f1"]
            .mean()
            .reset_index()
            .rename(columns={"macro_f1": "f1_set_b"})
        )

        if set_a.empty:
            logger.warning("No Set A ('%s') results found.", SET_A_MODE)
            return pd.DataFrame(columns=["dataset", "model", "f1_set_a", "f1_set_b", "delta_f1_abs"])

        if set_b.empty:
            logger.warning("No Set B ('%s') results found.", SET_B_MODE)
            return pd.DataFrame(columns=["dataset", "model", "f1_set_a", "f1_set_b", "delta_f1_abs"])

        merged = set_a.merge(set_b, on=["dataset", "model"], how="inner")

        if merged.empty:
            logger.warning(
                "No (dataset, model) pairs have both Set A and Set B results. "
                "Cannot compute robustness scores."
            )
            return pd.DataFrame(columns=["dataset", "model", "f1_set_a", "f1_set_b", "delta_f1_abs"])

        merged["delta_f1_abs"] = (merged["f1_set_a"] - merged["f1_set_b"]).abs()
        result = merged.sort_values("delta_f1_abs", ascending=False).reset_index(drop=True)

        logger.info(
            "Computed |ΔF1(A-B)| for %d (dataset, model) pairs.", len(result)
        )
        return result


def plot_robustness_heatmap(delta_df: pd.DataFrame, output_path: Path) -> None:
    """Produce a dataset × model heatmap of |ΔF1(A-B)| values.

    Requirements: 9.5, 7.5
    """
    if delta_df.empty:
        logger.warning("No robustness data -- skipping heatmap.")
        return

    pivot = delta_df.pivot_table(
        index="dataset",
        columns="model",
        values="delta_f1_abs",
    )

    # Shorten dataset names for readability
    def shorten(name: str) -> str:
        parts = name.split("/")
        return parts[-1] if len(parts) > 1 else name

    pivot.index = [shorten(d) for d in pivot.index]

    fig_height = max(4, len(pivot.index) * 0.6)
    fig_width = max(6, len(pivot.columns) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "|ΔF1(A-B)|"},
        vmin=0.0,
    )
    ax.set_title("|ΔF1(A-B)| Robustness Heatmap (Dataset × Model)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved robustness heatmap -> %s", output_path)


def write_robustness_summary(
    delta_df: pd.DataFrame,
    output_path: Path,
    avg_delta: float,
    threshold: float = ROBUSTNESS_THRESHOLD,
) -> None:
    """Write a Markdown summary of robustness findings.

    Includes average |ΔF1(A-B)|, robustness verdict, and the highest/lowest
    variance (dataset, model) pairs.

    Requirements: 9.6
    """
    lines = [
        "# Robustness Analysis Summary",
        "",
        "## Overview",
        "",
        f"- **Average |ΔF1(A-B)|**: {avg_delta:.4f}",
        f"- **Robustness threshold**: {threshold:.2f}",
    ]

    if np.isnan(avg_delta):
        verdict = "⚠️ **No data available** — Set A and/or Set B results are missing."
    elif avg_delta < threshold:
        verdict = (
            f"✅ Results are **robust to description phrasing** "
            f"(avg |ΔF1| = {avg_delta:.4f} < {threshold:.2f})."
        )
    else:
        verdict = (
            f"⚠️ Results are **NOT robust to description phrasing** "
            f"(avg |ΔF1| = {avg_delta:.4f} ≥ {threshold:.2f})."
        )

    lines += [
        f"- **Verdict**: {verdict}",
        "",
        f"Analysis based on **{len(delta_df)} (dataset, model) pairs**.",
        "",
    ]

    if not delta_df.empty:
        lines += [
            "## Highest Variance Pairs (top 5)",
            "",
            "| Dataset | Model | |ΔF1(A-B)| |",
            "|---------|-------|-----------|",
        ]
        top5 = delta_df.nlargest(5, "delta_f1_abs")
        for _, row in top5.iterrows():
            lines.append(f"| {row['dataset']} | {row['model']} | {row['delta_f1_abs']:.4f} |")

        lines += [
            "",
            "## Lowest Variance Pairs (top 5)",
            "",
            "| Dataset | Model | |ΔF1(A-B)| |",
            "|---------|-------|-----------|",
        ]
        bottom5 = delta_df.nsmallest(5, "delta_f1_abs")
        for _, row in bottom5.iterrows():
            lines.append(f"| {row['dataset']} | {row['model']} | {row['delta_f1_abs']:.4f} |")

        lines += [
            "",
            "## Per-Dataset Average |ΔF1(A-B)|",
            "",
            "| Dataset | Avg |ΔF1(A-B)| |",
            "|---------|--------------|",
        ]
        per_dataset = (
            delta_df.groupby("dataset")["delta_f1_abs"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        for _, row in per_dataset.iterrows():
            lines.append(f"| {row['dataset']} | {row['delta_f1_abs']:.4f} |")

        lines += [
            "",
            "## Per-Model Average |ΔF1(A-B)|",
            "",
            "| Model | Avg |ΔF1(A-B)| |",
            "|-------|--------------|",
        ]
        per_model = (
            delta_df.groupby("model")["delta_f1_abs"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        for _, row in per_model.iterrows():
            lines.append(f"| {row['model']} | {row['delta_f1_abs']:.4f} |")

    lines += ["", "---", "_Generated by `scripts/analyze_robustness.py`_", ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved robustness summary -> %s", output_path)


def run_analysis(
    results_dir: str = "results",
    output_dir: str = "reports/robustness",
) -> pd.DataFrame:
    """Run the full robustness analysis pipeline.

    Args:
        results_dir: Directory containing *_metrics.json result files.
        output_dir:  Directory to write output CSV, PNG, and Markdown.

    Returns:
        DataFrame with robustness scores (dataset, model, delta_f1_abs).

    Requirements: 7.2, 7.3, 9.3, 9.4, 9.5, 9.6, 7.5, 7.6
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv = output_path / "robustness_scores.csv"
    output_png = output_path / "robustness_heatmap.png"
    output_md = output_path / "robustness_summary.md"

    # --- Load data (Requirements 7.2) ---
    df = load_macro_f1_from_results(results_path)

    # Log input files used
    if not df.empty:
        set_a_files = df[df["label_mode"] == SET_A_MODE]["source_file"].unique()
        set_b_files = df[df["label_mode"] == SET_B_MODE]["source_file"].unique()
        logger.info("Set A input files (%d): %s", len(set_a_files), list(set_a_files))
        logger.info("Set B input files (%d): %s", len(set_b_files), list(set_b_files))

    if df.empty:
        logger.error(
            "No result data loaded from '%s'. "
            "Ensure *_metrics.json files with label_mode='%s' or '%s' are present.",
            results_path,
            SET_A_MODE,
            SET_B_MODE,
        )
        # Still produce empty outputs so downstream scripts don't break
        empty_df = pd.DataFrame(columns=["dataset", "model", "delta_f1_abs"])
        empty_df.to_csv(output_csv, index=False)
        logger.info("Saved empty robustness scores -> %s", output_csv)
        write_robustness_summary(empty_df, output_md, avg_delta=float("nan"))
        return empty_df

    # --- Compute robustness scores ---
    analyzer = RobustnessAnalyzer(df)
    delta_df = analyzer.compute_delta_f1()

    if delta_df.empty:
        logger.warning(
            "No (dataset, model) pairs with both Set A and Set B results found. "
            "Producing empty outputs."
        )
        empty_df = pd.DataFrame(columns=["dataset", "model", "delta_f1_abs"])
        empty_df.to_csv(output_csv, index=False)
        logger.info("Saved empty robustness scores -> %s", output_csv)
        write_robustness_summary(empty_df, output_md, avg_delta=float("nan"))
        return empty_df

    # --- Compute average |ΔF1| (Requirements 9.4) ---
    avg_delta = float(delta_df["delta_f1_abs"].mean())
    logger.info(
        "Average |ΔF1(A-B)| across all pairs: %.4f (threshold=%.2f) → %s",
        avg_delta,
        ROBUSTNESS_THRESHOLD,
        "ROBUST" if avg_delta < ROBUSTNESS_THRESHOLD else "NOT ROBUST",
    )

    # --- Save CSV (Requirements 7.6, 9.5) ---
    scores_csv = delta_df[["dataset", "model", "delta_f1_abs"]].copy()
    scores_csv.to_csv(output_csv, index=False)
    logger.info("Saved robustness scores -> %s", output_csv)

    # --- Save heatmap (Requirements 7.5, 9.5) ---
    plot_robustness_heatmap(delta_df, output_png)

    # --- Save summary Markdown (Requirements 9.6) ---
    write_robustness_summary(delta_df, output_md, avg_delta=avg_delta)

    # --- Log output files produced (Requirements 7.6) ---
    logger.info(
        "Output files produced: %s, %s, %s",
        output_csv,
        output_png,
        output_md,
    )

    # --- Print summary ---
    print(f"\nRobustness Analysis Results")
    print(f"  Average |ΔF1(A-B)|: {avg_delta:.4f}")
    print(
        f"  Verdict: {'ROBUST' if avg_delta < ROBUSTNESS_THRESHOLD else 'NOT ROBUST'} "
        f"(threshold={ROBUSTNESS_THRESHOLD:.2f})"
    )
    print(f"\nTop 5 highest-variance pairs:")
    print(delta_df.head(5)[["dataset", "model", "delta_f1_abs"]].to_string(index=False))

    return scores_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute robustness scores |ΔF1(A-B)| for each (dataset, model) pair "
            "by comparing description_set_a and description_set_b Macro-F1 results."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing *_metrics.json result files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/robustness",
        help="Output directory for CSV, PNG, and Markdown (default: reports/robustness)",
    )
    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
