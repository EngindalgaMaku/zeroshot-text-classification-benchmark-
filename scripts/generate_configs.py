#!/usr/bin/env python3
"""Generate YAML config files for all dataset × model combinations.

Modes:
  label_formulation  — 9 datasets × 7 models × multi_description (63 files)
  robustness         — 9 datasets × 7 models × {description_set_a, description_set_b} (126 files)
"""

import argparse
from pathlib import Path

import yaml

DATASETS = [
    {
        "key": "ag_news",
        "name": "ag_news",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "dbpedia_14",
        "name": "dbpedia_14",
        "split": "test",
        "text_column": "content",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "yahoo_answers_topics",
        "name": "yahoo_answers_topics",
        "split": "test",
        "text_column": "best_answer",
        "label_column": "topic",
        "max_samples": 1000,
    },
    {
        "key": "banking77",
        "name": "banking77",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "zeroshot_twitter_financial_news_sentiment",
        "name": "zeroshot/twitter-financial-news-sentiment",
        "split": "validation",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "SetFit_20_newsgroups",
        "name": "SetFit/20_newsgroups",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 2000,
    },
    {
        "key": "imdb",
        "name": "imdb",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "sst2",
        "name": "sst2",
        "split": "validation",
        "text_column": "sentence",
        "label_column": "label",
        "max_samples": 1000,
    },
    {
        "key": "go_emotions",
        "name": "go_emotions",
        "split": "test",
        "text_column": "text",
        "label_column": "labels",
        "max_samples": 1000,
    },
]

MODELS = [
    {"key": "instructor", "hf_name": "hkunlp/instructor-large", "batch_size": 16},
    {"key": "bge", "hf_name": "BAAI/bge-m3", "batch_size": 16},
    {"key": "mpnet", "hf_name": "sentence-transformers/all-mpnet-base-v2", "batch_size": 16},
    {"key": "nomic", "hf_name": "nomic-ai/nomic-embed-text-v2-moe", "batch_size": 8},
    {"key": "e5", "hf_name": "intfloat/multilingual-e5-large", "batch_size": 8},
    {"key": "jina", "hf_name": "jinaai/jina-embeddings-v3", "batch_size": 8},
    {"key": "qwen3", "hf_name": "Qwen/Qwen3-Embedding-8B", "batch_size": 8},
]

ROBUSTNESS_LABEL_MODES = ["description_set_a", "description_set_b"]


def build_config(dataset: dict, model: dict) -> dict:
    return {
        "experiment_name": f"{dataset['key']}_{model['key']}_multi_description",
        "dataset": {
            "name": dataset["name"],
            "split": dataset["split"],
            "text_column": dataset["text_column"],
            "label_column": dataset["label_column"],
            "max_samples": dataset["max_samples"],
        },
        "task": {
            "type": "zero_shot_classification",
            "label_mode": "multi_description",
            "language": "en",
        },
        "models": {
            "biencoder": {
                "provider": "hf",
                "name": model["hf_name"],
            },
            "reranker": None,
        },
        "pipeline": {
            "mode": "biencoder",
            "normalize_embeddings": True,
            "batch_size": model["batch_size"],
        },
        "evaluation": {
            "metrics": ["accuracy", "macro_f1", "per_class_f1"],
        },
        "output": {
            "save_predictions": True,
            "save_metrics": True,
            "output_dir": "results/label_semantics",
        },
    }


def build_robustness_config(dataset: dict, model: dict, label_mode: str) -> dict:
    return {
        "experiment_name": f"{dataset['key']}_{model['key']}_{label_mode}",
        "dataset": {
            "name": dataset["name"],
            "split": dataset["split"],
            "text_column": dataset["text_column"],
            "label_column": dataset["label_column"],
            "max_samples": dataset["max_samples"],
        },
        "task": {
            "type": "zero_shot_classification",
            "label_mode": label_mode,
            "language": "en",
        },
        "models": {
            "biencoder": {
                "provider": "hf",
                "name": model["hf_name"],
            },
            "reranker": None,
        },
        "pipeline": {
            "mode": "biencoder",
            "normalize_embeddings": True,
            "batch_size": model["batch_size"],
        },
        "evaluation": {
            "metrics": ["accuracy", "macro_f1", "per_class_f1"],
        },
        "output": {
            "save_predictions": True,
            "save_metrics": True,
            "output_dir": "results/robustness",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML config files for dataset × model combinations."
    )
    parser.add_argument(
        "--mode",
        choices=["label_formulation", "robustness"],
        default="label_formulation",
        help="Generation mode: 'label_formulation' (63 multi_description files) or 'robustness' (126 description_set_a/b files)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write config files into (default depends on mode)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip)",
    )
    args = parser.parse_args()

    if args.mode == "robustness":
        output_dir = Path(args.output_dir) if args.output_dir else Path("experiments/robustness")
        output_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        skipped = 0

        for dataset in DATASETS:
            for model in MODELS:
                for label_mode in ROBUSTNESS_LABEL_MODES:
                    filename = f"exp_{dataset['key']}_{model['key']}_{label_mode}.yaml"
                    filepath = output_dir / filename

                    if filepath.exists() and not args.overwrite:
                        print(f"  SKIP  {filename}")
                        skipped += 1
                        continue

                    config = build_robustness_config(dataset, model, label_mode)
                    with open(filepath, "w", encoding="utf-8") as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

                    print(f"  CREATE {filename}")
                    created += 1

        total = created + skipped
        print(f"\nDone: {created} created, {skipped} skipped, {total} total.")

    else:  # label_formulation
        output_dir = Path(args.output_dir) if args.output_dir else Path("experiments/label_formulation")
        output_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        skipped = 0

        for dataset in DATASETS:
            for model in MODELS:
                filename = f"exp_{dataset['key']}_{model['key']}_multi_description.yaml"
                filepath = output_dir / filename

                if filepath.exists() and not args.overwrite:
                    print(f"  SKIP  {filename}")
                    skipped += 1
                    continue

                config = build_config(dataset, model)
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

                print(f"  CREATE {filename}")
                created += 1

        total = created + skipped
        print(f"\nDone: {created} created, {skipped} skipped, {total} total.")


if __name__ == "__main__":
    main()
