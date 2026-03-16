"""Label Separability Analyzer.

Computes cosine similarity-based separability scores for each dataset × label mode
combination, then correlates separability with average Macro-F1 from results/.

Usage:
    python scripts/analyze_separability.py [--model MODEL_NAME] [--results-dir DIR]

Output:
    reports/separability/separability_scores.csv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders import BiEncoder
from src.labels import LABEL_SETS, build_multi_description_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Label mode → LABEL_SETS key mapping
LABEL_MODE_KEYS = {
    "L1": "name_only",
    "L2": "description",
    "L3": "multi_description",
}

# Default lightweight model
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LabelSeparabilityAnalyzer:
    """Computes label separability scores for datasets across label modes."""

    def __init__(self, encoder: BiEncoder):
        self.encoder = encoder

    def compute_similarity_matrix(
        self,
        label_dict: Dict[int, List[str]],
        label_mode_key: str,
    ) -> Optional[np.ndarray]:
        """Compute cosine similarity matrix between all label embeddings.

        For L3 (multi_description), embeddings are mean-pooled across descriptions.
        For L1/L2, each label has a single text; encode directly.

        Args:
            label_dict: Dict mapping label_id -> list of text strings
            label_mode_key: One of 'name_only', 'description', 'multi_description'

        Returns:
            Cosine similarity matrix of shape (n_labels, n_labels), or None if
            label count <= 1.
        """
        n_labels = len(label_dict)
        if n_labels <= 1:
            return None

        if label_mode_key == "multi_description":
            # Mean-pool multiple descriptions per label
            embeddings, _ = build_multi_description_embeddings(
                label_dict,
                self.encoder,
                normalize=True,
                batch_size=32,
            )
        else:
            # Single text per label — take the first (and only) element
            sorted_ids = sorted(label_dict.keys())
            texts = [label_dict[lid][0] for lid in sorted_ids]
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                normalize=True,
                show_progress=False,
                text_type="label",
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)

        sim_matrix = cosine_similarity(embeddings)
        return sim_matrix

    def compute_separability_score(self, sim_matrix: np.ndarray) -> float:
        """Compute separability score as mean of off-diagonal cosine similarities.

        A lower score means labels are more separable (less similar to each other).

        Args:
            sim_matrix: Square cosine similarity matrix (n x n)

        Returns:
            Mean of all off-diagonal elements.
        """
        n = sim_matrix.shape[0]
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = sim_matrix[mask]
        return float(off_diagonal.mean())

    def analyze_dataset(
        self,
        dataset_name: str,
        label_mode: str,
        label_mode_key: str,
    ) -> Optional[float]:
        """Compute separability score for one dataset × label mode.

        Returns None and logs a warning if label count <= 1.
        """
        if dataset_name not in LABEL_SETS:
            logger.warning("Dataset '%s' not found in LABEL_SETS — skipping.", dataset_name)
            return None

        if label_mode_key not in LABEL_SETS[dataset_name]:
            logger.warning(
                "Label mode '%s' not found for dataset '%s' — skipping.",
                label_mode_key,
                dataset_name,
            )
            return None

        label_dict = LABEL_SETS[dataset_name][label_mode_key]
        n_labels = len(label_dict)

        if n_labels <= 1:
            logger.warning(
                "Dataset '%s' (mode=%s) has %d label(s) — skipping (need > 1).",
                dataset_name,
                label_mode,
                n_labels,
            )
            return None

        logger.info(
            "Computing separability for '%s' | %s (%d labels)...",
            dataset_name,
            label_mode,
            n_labels,
        )
        sim_matrix = self.compute_similarity_matrix(label_dict, label_mode_key)
        if sim_matrix is None:
            return None

        score = self.compute_separability_score(sim_matrix)
        logger.info("  → separability_score = %.4f", score)
        return score


def load_macro_f1_from_results(results_dir: Path) -> pd.DataFrame:
    """Read Macro-F1 values from all JSON result files in results_dir.

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


def map_label_mode_to_code(label_mode: str) -> Optional[str]:
    """Map LABEL_SETS key to L1/L2/L3 code."""
    mapping = {
        "name_only": "L1",
        "description": "L2",
        "multi_description": "L3",
    }
    return mapping.get(label_mode)


def compute_pearson_correlation(
    sep_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    """Compute and log Pearson correlation between separability_score and avg Macro-F1.

    Merges on (dataset, label_mode) after normalising label_mode to L1/L2/L3 codes.
    """
    if results_df.empty:
        logger.warning("No result data available — skipping Pearson correlation.")
        return

    # Average Macro-F1 across models per (dataset, label_mode)
    results_df = results_df.copy()
    results_df["label_mode_code"] = results_df["label_mode"].apply(map_label_mode_to_code)
    results_df = results_df.dropna(subset=["label_mode_code"])

    avg_f1 = (
        results_df.groupby(["dataset", "label_mode_code"])["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"label_mode_code": "label_mode", "macro_f1": "avg_macro_f1"})
    )

    merged = sep_df.merge(avg_f1, on=["dataset", "label_mode"], how="inner")

    if len(merged) < 3:
        logger.warning(
            "Only %d overlapping (dataset, label_mode) pairs found — "
            "Pearson correlation requires at least 3 data points.",
            len(merged),
        )
        return

    r, p_value = pearsonr(merged["separability_score"], merged["avg_macro_f1"])
    logger.info(
        "Pearson correlation (separability_score vs avg Macro-F1): "
        "r=%.4f, p=%.4f (n=%d)",
        r,
        p_value,
        len(merged),
    )
    print(
        f"\nPearson correlation (separability_score vs avg Macro-F1): "
        f"r={r:.4f}, p={p_value:.4f} (n={len(merged)})\n"
    )


def run_analysis(
    model_name: str = DEFAULT_MODEL,
    results_dir: str = "results",
    output_dir: str = "reports/separability",
) -> pd.DataFrame:
    """Run the full separability analysis pipeline.

    Args:
        model_name: Sentence-transformer model for encoding labels.
        results_dir: Directory containing *_metrics.json result files.
        output_dir: Directory to write output CSV.

    Returns:
        DataFrame with separability scores.
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_csv = output_path / "separability_scores.csv"

    logger.info("Initialising encoder: %s", model_name)
    encoder = BiEncoder(model_name)
    analyzer = LabelSeparabilityAnalyzer(encoder)

    # Compute separability for every dataset × label mode
    rows = []
    for dataset_name in sorted(LABEL_SETS.keys()):
        for label_mode, label_mode_key in LABEL_MODE_KEYS.items():
            score = analyzer.analyze_dataset(dataset_name, label_mode, label_mode_key)
            if score is not None:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "label_mode": label_mode,
                        "separability_score": score,
                    }
                )

    if not rows:
        logger.error("No separability scores computed — check LABEL_SETS and encoder.")
        return pd.DataFrame(columns=["dataset", "label_mode", "separability_score"])

    sep_df = pd.DataFrame(rows)

    # Sort by separability_score (ascending = more separable first)
    sep_df = sep_df.sort_values("separability_score", ascending=True).reset_index(drop=True)

    # Save CSV
    sep_df.to_csv(output_csv, index=False)
    logger.info("Saved separability scores → %s", output_csv)

    # Load results and compute Pearson correlation
    results_df = load_macro_f1_from_results(results_path)
    compute_pearson_correlation(sep_df, results_df)

    # Print summary table
    print("\nSeparability Scores (sorted by score):")
    print(sep_df.to_string(index=False))

    return sep_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute label separability scores for all datasets and label modes."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing *_metrics.json result files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/separability",
        help="Output directory for CSV report (default: reports/separability)",
    )
    args = parser.parse_args()

    run_analysis(
        model_name=args.model,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
