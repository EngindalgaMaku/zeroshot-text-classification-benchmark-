#!/usr/bin/env python3
"""
Create missing label formulation config files
"""

import yaml
from pathlib import Path

# Dataset configurations
datasets = [
    ("ag_news", "text", "label", "test", 1000),
    ("banking77", "text", "label", "test", 1000),
    ("go_emotions", "text", "labels", "test", 1000),
]

# Model configurations
models = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet"),
    ("BAAI/bge-m3", "bge"),
    ("intfloat/multilingual-e5-large", "e5"),
    ("Qwen/Qwen3-Embedding-8B", "qwen3"),
    ("jinaai/jina-embeddings-v3", "jina"),
    ("hkunlp/instructor-large", "instructor"),
    ("nomic-ai/nomic-embed-text-v2-moe", "nomic"),
]

# Label modes
label_modes = ["name_only", "description"]

# Create output directory
output_dir = Path("experiments/label_formulation")
output_dir.mkdir(parents=True, exist_ok=True)

created_count = 0
skipped_count = 0

for ds_name, text_col, label_col, split, max_samples in datasets:
    ds_clean = ds_name.replace("/", "_").replace("-", "_")
    
    for model_name, model_short in models:
        for label_mode in label_modes:
            # Create experiment name
            exp_name = f"{ds_clean}_{model_short}_{label_mode}"
            config_path = output_dir / f"exp_{exp_name}.yaml"
            
            # Skip if already exists
            if config_path.exists():
                print(f"⏭️  Skip: {exp_name} (already exists)")
                skipped_count += 1
                continue
            
            # Create config
            config = {
                "experiment_name": exp_name,
                "dataset": {
                    "name": ds_name,
                    "split": split,
                    "text_column": text_col,
                    "label_column": label_col,
                    "max_samples": max_samples
                },
                "task": {
                    "type": "zero_shot_classification",
                    "label_mode": label_mode,
                    "language": "en"
                },
                "models": {
                    "biencoder": {"provider": "hf", "name": model_name},
                    "reranker": None
                },
                "pipeline": {
                    "mode": "biencoder",
                    "normalize_embeddings": True,
                    "batch_size": 16
                },
                "evaluation": {
                    "metrics": ["accuracy", "macro_f1", "per_class_f1"]
                },
                "output": {
                    "save_predictions": True,
                    "save_metrics": True,
                    "output_dir": "results/raw"
                }
            }
            
            # Write config
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Created: {exp_name}")
            created_count += 1

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Created: {created_count} new configs")
print(f"Skipped: {skipped_count} existing configs")
print(f"Total: {created_count + skipped_count} configs")
print(f"\n✅ Expected: 42 configs (3 datasets × 7 models × 2 modes)")