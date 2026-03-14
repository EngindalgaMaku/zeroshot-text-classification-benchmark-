"""Create INSTRUCTOR experiment configs for all 6 datasets."""

import yaml
from pathlib import Path

datasets = [
    ("ag_news", "text", "label", "test", 1000),
    ("SetFit/20_newsgroups", "text", "label", "test", 2000),
    ("dbpedia_14", "content", "label", "test", 1000),
    ("banking77", "text", "label", "test", 1000),
    ("yahoo_answers_topics", "best_answer", "topic", "test", 1000),
    ("zeroshot/twitter-financial-news-sentiment", "text", "label", "validation", 1000),
]

Path("experiments").mkdir(exist_ok=True)

for ds_name, text_col, label_col, split, max_samples in datasets:
    # Clean dataset name for experiment
    exp_name = ds_name.replace("/", "_").replace("-", "_")
    exp_name = f"{exp_name}_instructor"
    
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
            "biencoder": {
                "provider": "hf",
                "name": "hkunlp/instructor-large"
            },
            "reranker": None
        },
        "pipeline": {
            "mode": "biencoder",
            "normalize_embeddings": True
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
    
    config_path = f"experiments/{exp_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ {config_path}")

print(f"\n📊 Created {len(datasets)} INSTRUCTOR configs!")