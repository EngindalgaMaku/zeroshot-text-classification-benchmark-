"""Experiment runner."""

from typing import Dict, Any
from pathlib import Path
from src.data import load_text_classification_dataset, prepare_texts_and_labels
from src.labels import get_label_texts, flatten_label_texts
from src.encoders import BiEncoder
from src.pipeline import predict_biencoder
from src.metrics import compute_metrics, compute_confidence_metrics, analyze_errors
from src.utils import save_metrics, save_predictions, print_metrics_summary


def run_experiment(cfg: Dict[str, Any], skip_existing: bool = False):
    """Run a complete experiment.

    Args:
        cfg: Configuration dictionary
        skip_existing: If True, skip experiment if results already exist
    """
    exp_name = cfg["experiment_name"]
    output_dir = cfg["output"]["output_dir"]

    # Check if results already exist
    if skip_existing:
        metrics_file = Path(output_dir) / "raw" / f"{exp_name}_metrics.json"
        if metrics_file.exists():
            print("\n" + "=" * 70)
            print(f"⏭️  SKIPPING: {exp_name}")
            print(f"   Results already exist: {metrics_file}")
            print("=" * 70 + "\n")
            return None

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70 + "\n")

    # Load dataset
    dataset = load_text_classification_dataset(cfg)
    texts, y_true = prepare_texts_and_labels(
        dataset,
        cfg["dataset"]["text_column"],
        cfg["dataset"]["label_column"],
        dataset_name=cfg["dataset"]["name"],
    )

    # Get label texts
    ds_name = cfg["dataset"]["name"]
    label_mode = cfg["task"]["label_mode"]
    grouped_labels = get_label_texts(ds_name, label_mode)

    # Check for few-shot learning
    few_shot_config = cfg.get("pipeline", {}).get("few_shot", {})
    if few_shot_config.get("enabled", False):
        print("\n" + "=" * 70)
        print("FEW-SHOT LEARNING ENABLED")
        print("=" * 70)
        n_shots = few_shot_config.get("n_shots", 3)
        seed = few_shot_config.get("seed", 42)
        print(f"N-shots: {n_shots}")
        print(f"Seed: {seed}")

        # Import few-shot module
        from src.few_shot import setup_few_shot_labels

        # Load training data for few-shot examples
        print("\nLoading training data for few-shot examples...")
        train_cfg = cfg.copy()
        train_cfg["dataset"]["split"] = "train"
        train_cfg["dataset"]["max_samples"] = None

        train_dataset = load_text_classification_dataset(train_cfg)
        train_texts, train_labels = prepare_texts_and_labels(
            train_dataset,
            cfg["dataset"]["text_column"],
            cfg["dataset"]["label_column"],
            dataset_name=cfg["dataset"]["name"],
        )

        # Get class names
        class_names = [grouped_labels[i][0] for i in sorted(grouped_labels.keys())]

        # Convert grouped_labels to base descriptions
        base_descriptions = {k: v[0] for k, v in grouped_labels.items()}

        # Setup few-shot
        enhanced_descriptions, examples = setup_few_shot_labels(
            texts=train_texts,
            labels=train_labels,
            class_names=class_names,
            base_descriptions=base_descriptions,
            n_shots=n_shots,
            seed=seed,
        )

        # Update grouped_labels with enhanced descriptions
        grouped_labels = {k: [v] for k, v in enhanced_descriptions.items()}

        print("\n" + "=" * 70)
        print("FEW-SHOT SETUP COMPLETE")
        print("=" * 70 + "\n")

    print(f"Label mode: {label_mode}")
    print(f"Number of classes: {len(grouped_labels)}")
    print(f"Number of texts: {len(texts)}\n")

    # Initialize bi-encoder
    normalize = cfg["pipeline"].get("normalize_embeddings", True)
    biencoder_cfg = cfg["models"]["biencoder"]
    biencoder_name = biencoder_cfg["name"]
    biencoder_task = biencoder_cfg.get("task")
    allow_gte = biencoder_cfg.get("allow_gte", False)

    encoder = BiEncoder(
        biencoder_name,
        task=biencoder_task,
        allow_gte=allow_gte,
    )

    # Run pipeline (biencoder only)
    print("\nRunning biencoder pipeline\n")
    flat_texts, flat_ids = flatten_label_texts(grouped_labels)
    y_pred, confidences, _ = predict_biencoder(
        texts,
        flat_texts,
        flat_ids,
        encoder,
        normalize=normalize,
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred)
    confidence_metrics = compute_confidence_metrics(y_true, y_pred, confidences)
    metrics.update(confidence_metrics)

    # Add experiment metadata
    metrics["experiment_name"] = cfg["experiment_name"]
    metrics["dataset"] = ds_name
    metrics["label_mode"] = label_mode
    metrics["pipeline_mode"] = "biencoder"
    metrics["num_samples"] = len(texts)
    metrics["biencoder"] = encoder.model_name
    if biencoder_task is not None:
        metrics["biencoder_task"] = biencoder_task

    # Add few-shot metadata if enabled
    if few_shot_config.get("enabled", False):
        metrics["few_shot_enabled"] = True
        metrics["few_shot_n_shots"] = n_shots

    # Print results
    print_metrics_summary(metrics)

    # Analyze errors
    errors = analyze_errors(texts, y_true, y_pred, confidences, top_k=20)
    metrics["top_errors"] = errors

    # Save results
    output_dir = cfg["output"]["output_dir"]
    exp_name = cfg["experiment_name"]

    if cfg["output"].get("save_metrics", True):
        save_metrics(metrics, output_dir, exp_name)

    if cfg["output"].get("save_predictions", True):
        rows = []
        for text, yt, yp, conf in zip(texts, y_true, y_pred, confidences):
            rows.append(
                {
                    "text": text,
                    "y_true": yt,
                    "y_pred": yp,
                    "confidence": conf,
                    "correct": yt == yp,
                }
            )
        save_predictions(rows, output_dir, exp_name)

    print(f"\nExperiment complete: {exp_name}\n")

    return metrics