"""Label Embedding Trajectory Analysis (PCA / t-SNE).

Encodes L1 (name_only), L2 (description), and L3 (multi_description) label
texts for every dataset using each benchmark model, then projects the
embeddings to 2-D with PCA and t-SNE.

A "semantic trajectory" is drawn for each class: an arrow from the L1
centroid → L2 centroid → L3 centroid, showing how richer label descriptions
shift the embedding representation.

Outputs (one set per model):
  results/plots/trajectory/<model_key>_pca_trajectory.pdf  / .eps
  results/plots/trajectory/<model_key>_tsne_trajectory.pdf / .eps

Usage:
    python scripts/label_embedding_trajectory.py [--datasets DATASET ...]
                                                  [--models MODEL ...]
                                                  [--output-dir DIR]
                                                  [--no-tsne]
                                                  [--perplexity INT]

Requirements: 7.3, 7.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
# Publication styling
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Benchmark model registry
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    {"key": "bge",        "hf_name": "BAAI/bge-m3",                              "display": "BGE-M3"},
    {"key": "e5",         "hf_name": "intfloat/multilingual-e5-large",            "display": "E5-large"},
    {"key": "instructor", "hf_name": "hkunlp/instructor-large",                   "display": "INSTRUCTOR"},
    {"key": "jina_v5",    "hf_name": "jinaai/jina-embeddings-v3",                 "display": "Jina v5"},
    {"key": "mpnet",      "hf_name": "sentence-transformers/all-mpnet-base-v2",   "display": "MPNet"},
    {"key": "nomic",      "hf_name": "nomic-ai/nomic-embed-text-v2-moe",          "display": "Nomic-MoE"},
    {"key": "qwen3",      "hf_name": "Qwen/Qwen3-Embedding-8B",                   "display": "Qwen3"},
]

# Label level → LABEL_SETS key
LEVEL_KEYS = {
    "L1": "name_only",
    "L2": "description",
    "L3": "multi_description",
}

# Datasets available in LABEL_SETS (use all by default)
ALL_DATASETS = sorted(LABEL_SETS.keys())

# Colour palette for datasets (up to 12 distinct colours)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

# Marker styles for label levels
LEVEL_MARKERS = {"L1": "o", "L2": "s", "L3": "^"}
LEVEL_SIZES   = {"L1": 60,  "L2": 60,  "L3": 60}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def encode_level(
    label_dict: Dict[int, List[str]],
    level_key: str,
    encoder: BiEncoder,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[int]]:
    """Encode labels for one level, returning (embeddings, label_ids).

    For L3 (multi_description) the three descriptions per class are mean-pooled
    to produce a single centroid embedding per class.
    For L1/L2 each class has exactly one text.

    Args:
        label_dict: Dict mapping label_id -> list[str]
        level_key:  LABEL_SETS key ('name_only', 'description', 'multi_description')
        encoder:    BiEncoder instance
        batch_size: Encoding batch size

    Returns:
        embeddings: np.ndarray of shape (n_classes, dim)
        label_ids:  list of class IDs in sorted order
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


def collect_embeddings(
    datasets: List[str],
    encoder: BiEncoder,
    batch_size: int = 32,
) -> Dict[str, Dict[str, Tuple[np.ndarray, List[int]]]]:
    """Encode all datasets × all levels.

    Returns:
        {dataset_name: {level ('L1'|'L2'|'L3'): (embeddings, label_ids)}}
    """
    result: Dict[str, Dict[str, Tuple[np.ndarray, List[int]]]] = {}

    for ds in datasets:
        if ds not in LABEL_SETS:
            logger.warning("Dataset '%s' not in LABEL_SETS — skipping.", ds)
            continue

        result[ds] = {}
        for level, level_key in LEVEL_KEYS.items():
            if level_key not in LABEL_SETS[ds]:
                logger.warning(
                    "Level '%s' (%s) missing for dataset '%s' — skipping.",
                    level, level_key, ds,
                )
                continue

            label_dict = LABEL_SETS[ds][level_key]
            logger.info("  Encoding %s | %s (%d classes)…", ds, level, len(label_dict))
            embs, ids = encode_level(label_dict, level_key, encoder, batch_size)
            result[ds][level] = (embs, ids)

    return result


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def stack_all_embeddings(
    data: Dict[str, Dict[str, Tuple[np.ndarray, List[int]]]],
) -> Tuple[np.ndarray, List[Tuple[str, str, int]]]:
    """Stack all embeddings into one matrix for joint projection.

    Returns:
        matrix: (N, dim) array of all embeddings
        meta:   list of (dataset, level, label_id) for each row
    """
    rows, meta = [], []
    for ds, levels in data.items():
        for level, (embs, ids) in levels.items():
            for emb, lid in zip(embs, ids):
                rows.append(emb)
                meta.append((ds, level, lid))
    return np.stack(rows, axis=0).astype(np.float32), meta


def apply_pca(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to 2-D with PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(matrix)


def apply_tsne(
    matrix: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Project embeddings to 2-D with t-SNE."""
    # t-SNE works better on normalised / whitened data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    # Cap perplexity to n_samples - 1
    effective_perplexity = min(perplexity, max(5.0, len(matrix) - 1))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        n_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(scaled)


def build_centroid_map(
    coords_2d: np.ndarray,
    meta: List[Tuple[str, str, int]],
) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    """Build {dataset: {level: {label_id: 2d_coord}}} from projected coords.

    Since each (dataset, level, label_id) combination appears exactly once
    (one centroid per class per level), this is a direct lookup.
    """
    centroid_map: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    for coord, (ds, level, lid) in zip(coords_2d, meta):
        centroid_map.setdefault(ds, {}).setdefault(level, {})[lid] = coord
    return centroid_map


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _dataset_short_name(ds: str) -> str:
    """Return a short display name for a dataset key."""
    mapping = {
        "ag_news": "AG News",
        "dbpedia_14": "DBpedia",
        "yahoo_answers_topics": "Yahoo",
        "banking77": "Banking77",
        "zeroshot/twitter-financial-news-sentiment": "Twitter Fin.",
        "SetFit/20_newsgroups": "20 News",
        "imdb": "IMDB",
        "sst2": "SST-2",
        "go_emotions": "GoEmotions",
    }
    return mapping.get(ds, ds.split("/")[-1])


def draw_trajectory_figure(
    centroid_map: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    coords_2d: np.ndarray,
    meta: List[Tuple[str, str, int]],
    model_display: str,
    method: str,
    datasets: List[str],
) -> plt.Figure:
    """Draw the semantic trajectory figure for one model and one projection method.

    Each dataset gets its own subplot. Within each subplot, arrows connect
    L1 → L2 → L3 centroids for every class.

    Args:
        centroid_map:  {dataset: {level: {label_id: 2d_coord}}}
        coords_2d:     All projected coordinates (N, 2)
        meta:          Metadata list aligned with coords_2d
        model_display: Human-readable model name for the title
        method:        'PCA' or 't-SNE'
        datasets:      Ordered list of dataset names to plot

    Returns:
        matplotlib Figure
    """
    active_datasets = [ds for ds in datasets if ds in centroid_map]
    n = len(active_datasets)
    if n == 0:
        raise ValueError("No datasets to plot.")

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Label Embedding Trajectories — {model_display} ({method})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for idx, ds in enumerate(active_datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        ds_levels = centroid_map.get(ds, {})
        ds_color = _PALETTE[idx % len(_PALETTE)]
        ds_short = _dataset_short_name(ds)

        # Scatter all points for this dataset (background context)
        ds_mask = [i for i, (d, _, _) in enumerate(meta) if d == ds]
        if ds_mask:
            ax.scatter(
                coords_2d[ds_mask, 0],
                coords_2d[ds_mask, 1],
                c=ds_color,
                alpha=0.15,
                s=20,
                zorder=1,
                linewidths=0,
            )

        # Draw trajectories per class
        all_class_ids = set()
        for level_data in ds_levels.values():
            all_class_ids.update(level_data.keys())

        for class_id in sorted(all_class_ids):
            pts = {}
            for level in ("L1", "L2", "L3"):
                if level in ds_levels and class_id in ds_levels[level]:
                    pts[level] = ds_levels[level][class_id]

            # Draw L1 → L2 arrow
            if "L1" in pts and "L2" in pts:
                _draw_arrow(ax, pts["L1"], pts["L2"], color=ds_color, alpha=0.7)

            # Draw L2 → L3 arrow
            if "L2" in pts and "L3" in pts:
                _draw_arrow(ax, pts["L2"], pts["L3"], color=ds_color, alpha=0.9,
                            linestyle="dashed")

        # Plot centroids per level
        for level, marker, size in [
            ("L1", "o", 80), ("L2", "s", 80), ("L3", "^", 80)
        ]:
            if level not in ds_levels:
                continue
            pts_arr = np.array(list(ds_levels[level].values()))
            ax.scatter(
                pts_arr[:, 0], pts_arr[:, 1],
                marker=marker,
                s=size,
                c=ds_color,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label=level if idx == 0 else "_nolegend_",
            )

        ax.set_title(ds_short, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{method} dim 1", fontsize=10)
        ax.set_ylabel(f"{method} dim 2", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.3, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend for label levels
    level_handles = [
        mpatches.Patch(color="gray", label="L1 (name only)"),
        mpatches.Patch(color="gray", label="L2 (description)", alpha=0.6),
        mpatches.Patch(color="gray", label="L3 (full description)", alpha=0.3),
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, label="L1 centroid"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markersize=8, label="L2 centroid"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=8, label="L3 centroid"),
        plt.Line2D([0], [0], color="gray", linewidth=1.5, label="L1→L2 arrow"),
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
                   label="L2→L3 arrow"),
    ]
    fig.legend(
        handles=marker_handles,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    return fig


def _draw_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str = "gray",
    alpha: float = 0.8,
    linestyle: str = "solid",
) -> None:
    """Draw an annotated arrow from start to end on ax."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            alpha=alpha,
            lw=1.2,
            linestyle=linestyle,
            mutation_scale=10,
        ),
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

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
# Per-model pipeline
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: dict,
    datasets: List[str],
    output_dir: Path,
    run_tsne: bool = True,
    tsne_perplexity: float = 30.0,
    batch_size: int = 32,
) -> None:
    """Run the full trajectory analysis for one model.

    Steps:
      1. Load the encoder.
      2. Encode all datasets × all levels.
      3. Stack embeddings and apply PCA (and optionally t-SNE).
      4. Build centroid maps and draw trajectory figures.
      5. Save PDF + EPS outputs.
    """
    key = model_cfg["key"]
    hf_name = model_cfg["hf_name"]
    display = model_cfg["display"]

    logger.info("=" * 60)
    logger.info("Model: %s (%s)", display, hf_name)
    logger.info("=" * 60)

    # Load encoder
    encoder = BiEncoder(hf_name)

    # Encode all datasets × levels
    data = collect_embeddings(datasets, encoder, batch_size=batch_size)

    if not data:
        logger.warning("No embeddings collected for model '%s' — skipping.", key)
        return

    # Stack into one matrix
    matrix, meta = stack_all_embeddings(data)
    logger.info("Total embeddings: %d × %d", matrix.shape[0], matrix.shape[1])

    # ---- PCA ----
    logger.info("Applying PCA…")
    pca_coords = apply_pca(matrix)
    pca_centroids = build_centroid_map(pca_coords, meta)

    fig_pca = draw_trajectory_figure(
        pca_centroids, pca_coords, meta, display, "PCA", datasets
    )
    save_figure(fig_pca, output_dir, f"{key}_pca_trajectory")
    plt.close(fig_pca)

    # ---- t-SNE (optional) ----
    if run_tsne:
        logger.info("Applying t-SNE (perplexity=%.1f)…", tsne_perplexity)
        tsne_coords = apply_tsne(matrix, perplexity=tsne_perplexity)
        tsne_centroids = build_centroid_map(tsne_coords, meta)

        fig_tsne = draw_trajectory_figure(
            tsne_centroids, tsne_coords, meta, display, "t-SNE", datasets
        )
        save_figure(fig_tsne, output_dir, f"{key}_tsne_trajectory")
        plt.close(fig_tsne)

    logger.info("Done with model '%s'.", key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label embedding trajectory visualizations (PCA / t-SNE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Dataset keys to include (default: all datasets in LABEL_SETS).",
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
        "--output-dir",
        default="results/plots/trajectory",
        help="Directory to write output figures.",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        help="Skip t-SNE projection (faster, PCA only).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter.",
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
    model_keys = args.models if args.models else None
    if model_keys:
        key_set = {m["key"] for m in BENCHMARK_MODELS}
        unknown_m = [k for k in model_keys if k not in key_set]
        if unknown_m:
            logger.error("Unknown model keys: %s", unknown_m)
            logger.error("Available: %s", [m["key"] for m in BENCHMARK_MODELS])
            sys.exit(1)
        models_to_run = [m for m in BENCHMARK_MODELS if m["key"] in model_keys]
    else:
        models_to_run = BENCHMARK_MODELS

    output_dir = Path(args.output_dir)

    logger.info("Datasets  : %s", datasets)
    logger.info("Models    : %s", [m["key"] for m in models_to_run])
    logger.info("Output dir: %s", output_dir)
    logger.info("t-SNE     : %s", not args.no_tsne)

    for model_cfg in models_to_run:
        run_model(
            model_cfg=model_cfg,
            datasets=datasets,
            output_dir=output_dir,
            run_tsne=not args.no_tsne,
            tsne_perplexity=args.perplexity,
            batch_size=args.batch_size,
        )

    logger.info("All models complete. Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
