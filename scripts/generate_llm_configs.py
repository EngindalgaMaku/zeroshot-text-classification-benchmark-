"""Generate experiment configs for LLM-generated label descriptions (L2 and L3).

This script creates YAML config files for all datasets with l2 and l3 label modes.
"""

from pathlib import Path

# Datasets and their basic configs (matching main experiments)
DATASETS = {
    "ag_news": {"split": "test", "max_samples": 1000},
    "dbpedia_14": {"split": "test", "max_samples": 1000},
    "yahoo_answers_topics": {"split": "test", "max_samples": 1000},
    "banking77": {"split": "test", "max_samples": 1000},
    "zeroshot/twitter-financial-news-sentiment": {"split": "test", "max_samples": 2000},
    "SetFit/20_newsgroups": {"split": "test", "max_samples": 2000},
    "imdb": {"split": "test", "max_samples": 2000},
    "sst2": {"split": "validation", "max_samples": 1000},
    "go_emotions": {"split": "test", "max_samples": 2000},
}

# Models to test - ACTIVE models only
MODELS = [
    "sentence-transformers/all-mpnet-base-v2",           # mpnet
    "BAAI/bge-small-en-v1.5",                           # bge
    "nomic-ai/nomic-embed-text-v1.5",                   # nomic
    "intfloat/e5-small-v2",                             # e5
    "hkunlp/instructor-base",                           # instructor
    "jinaai/jina-embeddings-v3",                        # jina_v5
    # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",           # REMOVED - not used anymore
]

def generate_configs():
    """Generate experiment configs for L1, L2, and L3 label modes."""
    
    output_dir = Path("experiments/llm_descriptions")
    output_dir.mkdir(exist_ok=True)
    
    configs_created = []
    
    for dataset_name, dataset_config in DATASETS.items():
        dataset_id = dataset_name.replace("/", "_")
        
        for model in MODELS:
            model_id = model.split("/")[-1]
            
            # L1 config (name_only)
            l1_name = f"{dataset_id}_{model_id}_l1"
            l1_config = {
                "experiment_name": l1_name,
                "dataset": {
                    "name": dataset_name,
                    "split": dataset_config["split"],
                    "text_column": "text",
                    "label_column": "label",
                },
                "task": {
                    "label_mode": "name_only",  # L1 - just label names
                },
                "models": {
                    "biencoder": {
                        "name": model,
                    }
                },
                "pipeline": {
                    "normalize_embeddings": True,
                },
                "output": {
                    "output_dir": "results/llm_descriptions",
                    "save_metrics": True,
                    "save_predictions": True,
                }
            }
            
            if "max_samples" in dataset_config:
                l1_config["dataset"]["max_samples"] = dataset_config["max_samples"]
            
            # Write L1 config
            import yaml
            l1_path = output_dir / f"{l1_name}.yaml"
            with open(l1_path, "w") as f:
                yaml.dump(l1_config, f, default_flow_style=False, sort_keys=False)
            configs_created.append(str(l1_path))
            
            # L2 config (single description)
            l2_name = f"{dataset_id}_{model_id}_l2"
            l2_config = {
                "experiment_name": l2_name,
                "dataset": {
                    "name": dataset_name,
                    "split": dataset_config["split"],
                    "text_column": "text",
                    "label_column": "label",
                },
                "task": {
                    "label_mode": "l2",  # LLM-generated single description
                },
                "models": {
                    "biencoder": {
                        "name": model,
                    }
                },
                "pipeline": {
                    "normalize_embeddings": True,
                },
                "output": {
                    "output_dir": "results/llm_descriptions",
                    "save_metrics": True,
                    "save_predictions": True,
                }
            }
            
            if "max_samples" in dataset_config:
                l2_config["dataset"]["max_samples"] = dataset_config["max_samples"]
            
            # Write L2 config
            import yaml
            l2_path = output_dir / f"{l2_name}.yaml"
            with open(l2_path, "w") as f:
                yaml.dump(l2_config, f, default_flow_style=False, sort_keys=False)
            configs_created.append(str(l2_path))
            
            # L3 config (multi-aspect descriptions with mean pooling)
            l3_name = f"{dataset_id}_{model_id}_l3"
            l3_config = l2_config.copy()
            l3_config["experiment_name"] = l3_name
            l3_config["task"] = {"label_mode": "l3"}  # LLM-generated multi-aspect
            
            # Write L3 config
            l3_path = output_dir / f"{l3_name}.yaml"
            with open(l3_path, "w") as f:
                yaml.dump(l3_config, f, default_flow_style=False, sort_keys=False)
            configs_created.append(str(l3_path))
    
    print(f"✅ Created {len(configs_created)} experiment configs:")
    print(f"   - {len(configs_created)//3} L1 configs (name_only)")
    print(f"   - {len(configs_created)//3} L2 configs (LLM single)")
    print(f"   - {len(configs_created)//3} L3 configs (LLM multi)")
    print(f"   - Output: {output_dir}/")
    
    return configs_created


if __name__ == "__main__":
    generate_configs()