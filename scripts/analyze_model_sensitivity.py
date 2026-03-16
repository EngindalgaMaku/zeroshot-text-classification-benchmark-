"""Model Sensitivity Analyzer.

Computes variance of Macro-F1 across label modes (L1/L2/L3) per model,
producing a sensitivity ranking table and bar chart.

Usage:
    python scripts/analyze_model_sensitivity.py [--results-dir DIR] [--output-dir DIR]

Output:
    reports/model_sensitivity/sensitivity_scores.csv
    reports/model_sensitivity/sensitivity_bar.png
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

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# label_mode string -> L1/L2/L3 code
LABEL_MODE_CODE: dict[str, str] = {
    "name_only": "L1",
    "description": "L2",
    "multi_description": "L3",
}


def load_macro_f1_from_results(results_dir: Path) -> pd.DataFrame:
    """Read Macro-F1 values from all *_metrics.json files under results_dir.

    Returns a DataFrame with columns: dataset, label_mode, model, macro_f1.
    Requirements: 7.2
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
            logger.warning("Missing fields in '%s' -- skipping.", fpath)
            continue

        records.append({
            "dataset": dataset,
            "label_mode": label_mode,
            "model": model,
            "macro_f1": float(macro_f1),
        })

    df = pd.DataFrame(records)
    logger.info("Loaded %d result records from %d files.", len(df), len(json_files))
    return df


class ModelSensitivityAnalyzer:
    """Computes label-mode sensitivity (F1 variance) for each model.

    For each model, averages Macro-F1 across all datasets per label mode,
    then computes variance across the (up to 3) label-mode averages.
    Models with missing label modes are included with a warning.

    Args:
        df: DataFrame with columns: dataset, label_mode, model, macro_f1

    Requirements: 6.1, 6.3, 6.5
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["label_mode_code"] = df["label_mode"].map(LABEL_MODE_CODE)
        unknown_modes = df[df["label_mode_code"].isna()]["label_mode"].unique()
        for mode in unknown_modes:
            logger.warning("Unknown label_mode '%s' -- skipping those rows.", mode)
        self.df = df.dropna(subset=["label_mode_code"])

    def compute_sensitivity(self) -> pd.DataFrame:
        """Compute per-model F1 variance across label modes.

        Steps:
        1. Average Macro-F1 across datasets per (model, label_mode_code).
        2. Compute variance of those averages across label modes.
        3. Warn if a model has fewer than 3 label modes.

        Returns:
            DataFrame with columns: model, avg_variance, n_label_modes
            Sorted by avg_variance descending (most sensitive first).

        Requirements: 6.1, 6.3, 6.5
        """
        # Step 1: average F1 per (model, label_mode_code) across datasets
        per_mode = (
            self.df.groupby(["model", "label_mode_code"])["macro_f1"]
            .mean()
            .reset_index()
            .rename(columns={"label_mode_code": "label_mode", "macro_f1": "avg_f1"})
        )

        rows = []
        for model, group in per_mode.groupby("model"):
            n_modes = len(group)
            if n_modes < 3:
                missing = set(["L1", "L2", "L3"]) - set(group["label_mode"].tolist())
                logger.warning(
                    "Model '%s' is missing label mode(s) %s -- computing variance from %d available mode(s).",
                    model,
                    missing,
                    n_modes,
                )
            variance = float(group["avg_f1"].var(ddof=0))  # population variance
            rows.append({"model": model, "avg_variance": variance, "n_label_modes": n_modes})

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        result = result.sort_values("avg_variance", ascending=False).reset_index(drop=True)
        return result


def plot_sensitivity_bar(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    """Produce a horizontal bar chart of model sensitivity scores.

    Models are sorted by avg_variance ascending so the most sensitive
    model appears at the top of the chart.

    Requirements: 6.4, 7.5
    """
    if sensitivity_df.empty:
        logger.warning("No sensitivity data -- skipping bar chart.")
        return

    # Sort ascending for horizontal bar (highest variance at top)
    plot_df = sensitivity_df.sort_values("avg_variance", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.55)))
    ax.barh(
        plot_df["model"],
        plot_df["avg_variance"],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xlabel("Avg Variance (F1 across label modes)")
    ax.set_ylabel("Model")
    ax.set_title("Model Sensitivity to Label Mode")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensitivity bar chart -> %s", output_path)


def run_analysis(
    results_dir: str = "results",
    output_dir: str = "reports/model_sensitivity",
) -> pd.DataFrame:
    """Run the full model sensitivity analysis pipeline.

    Args:
        results_dir: Directory containing *_metrics.json result files.
        output_dir:  Directory to write output CSV and PNG.

    Returns:
        DataFrame with sensitivity scores.

    Requirements: 7.2, 7.3, 7.6
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv = output_path / "sensitivity_scores.csv"
    output_png = output_path / "sensitivity_bar.png"

    # Load data
    df = load_macro_f1_from_results(results_path)

    if df.empty:
        logger.error(
            "No result data loaded from '%s'. "
            "Ensure *_metrics.json files are present.",
            results_path,
        )
        return pd.DataFrame(columns=["model", "avg_variance", "n_label_modes"])

    # Compute sensitivity
    analyzer = ModelSensitivityAnalyzer(df)
    sensitivity_df = analyzer.compute_sensitivity()

    if sensitivity_df.empty:
        logger.error("No sensitivity scores computed -- check result files.")
        return sensitivity_df

    # Save CSV (Requirements 7.6)
    sensitivity_df.to_csv(output_csv, index=False)
    logger.info("Saved sensitivity scores -> %s", output_csv)

    # Save bar chart (Requirements 7.5)
    plot_sensitivity_bar(sensitivity_df, output_png)

    # Log output files produced (Requirements 7.6)
    logger.info("Output files produced: %s, %s", output_csv, output_png)

    # Print summary
    print("\nModel Sensitivity Scores (sorted by avg_variance descending):")
    print(sensitivity_df.to_string(index=False))

    return sensitivity_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model sensitivity to label mode changes from Macro-F1 results."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing *_metrics.json result files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/model_sensitivity",
        help="Output directory for CSV and PNG (default: reports/model_sensitivity)",
    )
    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
