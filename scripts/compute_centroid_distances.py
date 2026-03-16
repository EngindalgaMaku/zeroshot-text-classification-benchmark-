"""Centroid Distance Metrics: Intra-Label Variance and Inter-Label Separation.

For each model and label level (L1/L2/L3), computes per-class embedding centroids,
then calculates:

  - intra_variance: mean cosine distance between the L1, L2, and L3 centroids of
    the *same* class (how much the representation shifts across label levels).
  - inter_separation: mean pairwise cosine distance between *different* class
    centroids at each label level (how well-separated the classes are).

Cosine distance is defined as  1 - cosine_similarity.

Output CSV columns:
    model, dataset, label_level, intra_variance, inter_separation

Usage:
    python scripts/compute_centroid_distances.py [--models MODEL ...] \\
                                                  [--datasets DATASET ...] \\
                                                  [--output PATH] \\
                                                  [--batch-size INT]

Requirements: 7.3, 7.4, 7.5
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders import BiEncoder
from src.labels import LABEL_SETS, build_multi_description_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark model registry (mirrors label_embedding_trajectory.py)
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    {"key": "bge",        "hf_name": "BAAI/bge-m3",                             "display": "BGE-M3"},
    {"key": "e5",         "hf_name": "intfloat/multilingual-e5-large",           "display": "E5-large"},
    {"key": "instructor", "hf_name": "hkunlp/instructor-large",                  "display": "INSTRUCTOR"},
    {"key": "jina_v5",    "hf_name": "jinaai/jina-embeddings-v3",                "display": "Jina v5"},
    {"key": "mpnet",      "hf_name": "sentence-transformers/all-mpnet-base-v2",  "display": "MPNet"},
    {"key": "nomic",      "hf_name": "nomic-ai/nomic-embed-text-v2-moe",         "display": "Nomic-MoE"},
    {"key": "qwen3",      "hf_name": "Qwen/Qwen3-Embedding-8B",                  "display": "Qwen3"},
]

# Label level → LABEL_SETS key
LEVEL_KEYS: Dict[str, str] = {
    "L1": "name_only",
    "L2": "description",
    "L3": "multi_description",
}

ALL_DATASETS = sorted(LABEL_SETS.keys())


# ---------------------------------------------------------------------------
# Embedding helpers (reused from label_embedding_trajectory.py)
# ---------------------------------------------------------------------------

def encode_level(
    label_dict: Dict[int, List[str]],
    level_key: str,
    encoder: BiEncoder,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[int]]:
    """Encode labels for one level, returning (centroids, label_ids).

    For L3 (multi_description) the three descriptions per class are mean-pooled
    to produce a single centroid embedding per class.
    For L1/L2 each class has exactly one text.

    Returns:
        centroids: np.ndarray of shape (n_classes, dim)
        label_ids: list of class IDs in sorted order
    """
    if level_key == "multi_description":
        return build_multi_description_embeddings(
            label_dict, encoder, normalize=True, batch_size=batch_size
        )

    sorted_ids = sorted(label_dict.keys())
    texts = [label_dict[lid][0] for lid in sorted_ids]
    embs = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize=True,
        show_progress=False,
        text_type="label",
    )
    return np.asarray(embs, dtype=np.float32), sorted_ids


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors: 1 - dot(a, b)."""
    return float(1.0 - np.dot(a, b))


def compute_intra_variance(
    centroids_by_level: Dict[str, Tuple[np.ndarray, List[int]]],
) -> Optional[float]:
    """Mean cosine distance between L1/L2/L3 centroids of the *same* class.

    For each class that appears in all three levels, compute the mean pairwise
    cosine distance among its three centroids (L1, L2, L3).  Then average over
    all classes.

    Returns None if fewer than two levels are available.
    """
    available_levels = [lv for lv in ("L1", "L2", "L3") if lv in centroids_by_level]
    if len(available_levels) < 2:
        return None

    # Build {label_id: {level: centroid_vector}}
    class_vecs: Dict[int, Dict[str, np.ndarray]] = {}
    for level in available_levels:
        embs, ids = centroids_by_level[level]
        for emb, lid in zip(embs, ids):
            class_vecs.setdefault(lid, {})[level] = emb

    # Only consider classes present in ALL available levels
    common_ids = [
        lid for lid, lvs in class_vecs.items()
        if all(lv in lvs for lv in available_levels)
    ]
    if not common_ids:
        return None

    per_class_dists = []
    for lid in common_ids:
        vecs = [class_vecs[lid][lv] for lv in available_levels]
        # Mean pairwise cosine distance among the level centroids for this class
        pair_dists = [
            cosine_distance(vecs[i], vecs[j])
            for i, j in combinations(range(len(vecs)), 2)
        ]
        per_class_dists.append(float(np.mean(pair_dists)))

    return float(np.mean(per_class_dists))


def compute_inter_separation(
    centroids: np.ndarray,
) -> Optional[float]:
    """Mean pairwise cosine distance between *different* class centroids.

    Args:
        centroids: (n_classes, dim) array of L2-normalised embeddings.

    Returns:
        Mean of all off-diagonal pairwise cosine distances, or None if < 2 classes.
    """
    n = len(centroids)
    if n < 2:
        return None

    # For L2-normalised vectors: cosine_distance = 1 - dot(a, b)
    # Gram matrix gives all dot products at once.
    gram = centroids @ centroids.T  # (n, n)
    # Off-diagonal elements only
    mask = ~np.eye(n, dtype=bool)
    mean_sim = float(gram[mask].mean())
    return float(1.0 - mean_sim)


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------

def compute_dataset_metrics(
    dataset: str,
    encoder: BiEncoder,
    batch_size: int = 32,
) -> List[dict]:
    """Compute centroid distance metrics for one dataset × all label levels.

    Returns a list of row dicts (one per label level) with keys:
        dataset, label_level, intra_variance, inter_separation
    """
    if dataset not in LABEL_SETS:
        logger.warning("Dataset '%s' not in LABEL_SETS — skipping.", dataset)
        return []

    # Encode all available levels
    centroids_by_level: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    for level, level_key in LEVEL_KEYS.items():
        if level_key not in LABEL_SETS[dataset]:
            logger.warning(
                "Level '%s' (%s) missing for dataset '%s' — skipping level.",
                level, level_key, dataset,
            )
            continue

        label_dict = LABEL_SETS[dataset][level_key]
        n_classes = len(label_dict)
        logger.info("  Encoding %s | %s (%d classes)…", dataset, level, n_classes)
        embs, ids = encode_level(label_dict, level_key, encoder, batch_size)
        centroids_by_level[level] = (embs, ids)

    if not centroids_by_level:
        return []

    # Intra-variance is a single value across all levels (cross-level metric)
    intra_var = compute_intra_variance(centroids_by_level)

    rows = []
    for level in ("L1", "L2", "L3"):
        if level not in centroids_by_level:
            continue

        embs, _ = centroids_by_level[level]
        inter_sep = compute_inter_separation(embs)

        rows.append({
            "dataset": dataset,
            "label_level": level,
            "intra_variance": intra_var,   # same value for all levels of this dataset
            "inter_separation": inter_sep,
        })

    return rows


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: dict,
    datasets: List[str],
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run centroid distance computation for one model across all datasets.

    Returns a DataFrame with columns:
        model, dataset, label_level, intra_variance, inter_separation
    """
    key = model_cfg["key"]
    hf_name = model_cfg["hf_name"]
    display = model_cfg["display"]

    logger.info("=" * 60)
    logger.info("Model: %s (%s)", display, hf_name)
    logger.info("=" * 60)

    encoder = BiEncoder(hf_name)

    all_rows = []
    for ds in datasets:
        logger.info("Dataset: %s", ds)
        rows = compute_dataset_metrics(ds, encoder, batch_size=batch_size)
        for row in rows:
            row["model"] = key
        all_rows.extend(rows)

    logger.info("Done with model '%s'. Collected %d rows.", key, len(all_rows))

    if not all_rows:
        return pd.DataFrame(
            columns=["model", "dataset", "label_level", "intra_variance", "inter_separation"]
        )

    df = pd.DataFrame(all_rows)[
        ["model", "dataset", "label_level", "intra_variance", "inter_separation"]
    ]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute centroid distance metrics (intra-label variance and "
            "inter-label separation) for all benchmark models and datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_KEY",
        help=(
            "Model keys to run (e.g. bge e5 mpnet). "
            "Default: all 7 benchmark models."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "Dataset keys to include. Default: all datasets in LABEL_SETS."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/centroid_distances.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve datasets
    datasets = args.datasets if args.datasets else ALL_DATASETS
    unknown_ds = [d for d in datasets if d not in LABEL_SETS]
    if unknown_ds:
        logger.error("Unknown datasets: %s", unknown_ds)
        logger.error("Available: %s", sorted(LABEL_SETS.keys()))
        sys.exit(1)

    # Resolve models
    model_key_set = {m["key"] for m in BENCHMARK_MODELS}
    if args.models:
        unknown_m = [k for k in args.models if k not in model_key_set]
        if unknown_m:
            logger.error("Unknown model keys: %s", unknown_m)
            logger.error("Available: %s", sorted(model_key_set))
            sys.exit(1)
        models_to_run = [m for m in BENCHMARK_MODELS if m["key"] in args.models]
    else:
        models_to_run = BENCHMARK_MODELS

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Datasets  : %s", datasets)
    logger.info("Models    : %s", [m["key"] for m in models_to_run])
    logger.info("Output    : %s", output_path)
    logger.info("Batch size: %d", args.batch_size)

    all_dfs = []
    for model_cfg in models_to_run:
        df = run_model(model_cfg, datasets, batch_size=args.batch_size)
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No results produced.")
        sys.exit(1)

    results = pd.concat(all_dfs, ignore_index=True)

    # Append to existing CSV if present, otherwise create new
    if output_path.exists():
        existing = pd.read_csv(output_path)
        # Drop rows that will be overwritten (same model + dataset + label_level)
        key_cols = ["model", "dataset", "label_level"]
        new_keys = set(zip(results["model"], results["dataset"], results["label_level"]))
        mask = existing.apply(
            lambda r: (r["model"], r["dataset"], r["label_level"]) not in new_keys,
            axis=1,
        )
        results = pd.concat([existing[mask], results], ignore_index=True)

    results.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(results), output_path)

    # Print summary
    print("\nCentroid Distance Metrics Summary:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()