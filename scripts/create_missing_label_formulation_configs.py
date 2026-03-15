#!/usr/bin/env python3
"""
Create missing label formulation config files for ALL 9 datasets
WITH CORRECT SAMPLE SIZES per documentation
"""

import yaml
from pathlib import Path

# ALL 9 Dataset configurations WITH CORRECT SAMPLE SIZES
datasets = [
    ("ag_news", "text", "label", "test", 1000),
    ("dbpedia_14", "content", "label", "test", 1000),
    ("yahoo_answers_topics", "best_answer", "topic", "test", 1000),
    ("banking77", "text", "label", "test", 1000),
    ("zeroshot/twitter-financial-news-sentiment", "text", "label", "validation", 1000),
    ("SetFit/20_newsgroups", "text", "label", "test", 2000),  # SPECIAL: 2000 samples
    ("go_emotions", "text", "labels", "test", 1000),
    ("imdb", "text", "label", "test", 1000),
    ("sst2", "sentence", "label", "validation", 1000),
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
updated_count = 0
skipped_count = 0

for ds_name, text_col, label_col, split, max_samples in datasets:
    ds_clean = ds_name.replace("/", "_").replace("-", "_")
    
    for model_name, model_short in models:
        for label_mode in label_modes:
            # Create experiment name
            exp_name = f"{ds_clean}_{model_short}_{label_mode}"
            config_path = output_dir / f"exp_{exp_name}.yaml"
            
            # Check if exists and has wrong sample size
            if config_path.exists():
                with open(config_path, 'r') as f:
                    existing_config = yaml.safe_load(f)
                
                existing_samples = existing_config.get('dataset', {}).get('max_samples', 0)
                
                if existing_samples != max_samples:
                    print(f"🔄 Update: {exp_name} ({existing_samples} → {max_samples})")
                    updated_count += 1
                else:
                    print(f"⏭️  Skip: {exp_name} (already correct)")
                    skipped_count += 1
                    continue
            else:
                print(f"✅ Create: {exp_name}")
                created_count += 1
            
            # Create/update config
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

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Created: {created_count} new configs")
print(f"Updated: {updated_count} configs (corrected sample sizes)")
print(f"Skipped: {skipped_count} existing configs (already correct)")
print(f"Total: {created_count + updated_count + skipped_count} configs")
print(f"\n✅ Expected: 126 configs (9 datasets × 7 models × 2 modes)")
print(f"\n📊 Sample sizes per documentation:")
print(f"  • Most datasets: 1,000 samples")
print(f"  • 20 Newsgroups: 2,000 samples (harder task)")