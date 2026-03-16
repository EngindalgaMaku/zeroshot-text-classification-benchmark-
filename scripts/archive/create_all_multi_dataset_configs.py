"""
Create all missing experiment configs for 9 datasets × 7 models = 63 experiments
"""
import yaml
from pathlib import Path

# 9 datasets
datasets = [
    ("ag_news", "text", "label", "test", 1000),
    ("dbpedia_14", "content", "label", "test", 1000),
    ("yahoo_answers_topics", "best_answer", "topic", "test", 1000),
    ("banking77", "text", "label", "test", 1000),
    ("zeroshot/twitter-financial-news-sentiment", "text", "label", "validation", 1000),
    ("SetFit/20_newsgroups", "text", "label", "test", 1000),
    ("go_emotions", "text", "labels", "test", 1000),
    ("imdb", "text", "label", "test", 1000),
    ("sst2", "sentence", "label", "validation", 1000),
]

# 7 models
models = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet"),
    ("BAAI/bge-m3", "bge"),
    ("intfloat/multilingual-e5-large", "e5"),
    ("Qwen/Qwen3-Embedding-8B", "qwen3"),
    ("jinaai/jina-embeddings-v3", "jina_v3"),
    ("hkunlp/instructor-large", "instructor"),
    ("nomic-ai/nomic-embed-text-v2-moe", "nomic"),
]

Path("experiments").mkdir(exist_ok=True)

created = 0
skipped = 0

for ds_name, text_col, label_col, split, max_samples in datasets:
    for model_name, model_short in models:
        # Clean dataset name for filename
        ds_clean = ds_name.replace("/", "_").replace("-", "_")
        
        # Experiment name
        exp_name = f"exp_{ds_clean}_{model_short}"
        config_path = Path(f"experiments/{exp_name}.yaml")
        
        # Skip if exists
        if config_path.exists():
            skipped += 1
            print(f"⏭️  {exp_name}.yaml (already exists)")
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
                "label_mode": "description",
                "language": "en"
            },
            "models": {
                "biencoder": {"provider": "hf", "name": model_name},
                "reranker": None
            },
            "pipeline": {
                "mode": "biencoder",
                "normalize_embeddings": True,
                "batch_size": 32  # GPU ile daha büyük batch
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
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        created += 1
        print(f"✅ {exp_name}.yaml")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Created: {created}")
print(f"Skipped (already exists): {skipped}")
print(f"Total: {created + skipped}")
print(f"Expected: {len(datasets) * len(models)} (9 datasets × 7 models)")
print(f"{'='*70}")