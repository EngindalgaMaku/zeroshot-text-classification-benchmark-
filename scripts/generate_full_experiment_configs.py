"""Generate 189 experiment YAML configs for label description experiments.

9 datasets × 7 models × 3 modes = 189 experiments
"""

import os
from pathlib import Path

# Dataset configurations (name: (text_column, label_column, split))
DATASETS = {
    "ag_news": ("text", "label", "test"),
    "yahoo_answers_topics": ("best_answer", "topic", "test"),
    "SetFit/20_newsgroups": ("text", "label", "test"),
    "dbpedia_14": ("content", "label", "test"),
    "banking77": ("text", "label", "test"),
    "imdb": ("text", "label", "test"),
    "sst2": ("sentence", "label", "validation"),
    "zeroshot/twitter-financial-news-sentiment": ("text", "label", "test"),
    "go_emotions": ("text", "labels", "test"),
}

# Model configurations (key: (full_name, batch_size))
MODELS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 32),
    "e5": ("intfloat/multilingual-e5-large", 32),
    "bge": ("BAAI/bge-large-en-v1.5", 32),
    "nomic": ("nomic-ai/nomic-embed-text-v1", 32),
    "jina": ("jinaai/jina-embeddings-v3", 32),
    "instructor": ("hkunlp/instructor-large", 16),
    "qwen3": ("Qwen/Qwen3-Embedding", 32),
}

# Label modes
MODES = ["name_only", "l2", "l3"]

# Template for experiment YAML
YAML_TEMPLATE = """experiment_name: full_{dataset_short}_{model}_{mode}
dataset:
  name: {dataset_name}
  split: {split}
  text_column: {text_column}
  label_column: {label_column}
  max_samples: null
task:
  type: zero_shot_classification
  label_mode: {mode}
  language: en
models:
  biencoder:
    provider: hf
    name: {model_name}
  reranker: null
pipeline:
  mode: biencoder
  normalize_embeddings: true
  batch_size: {batch_size}
evaluation:
  metrics:
  - accuracy
  - macro_f1
  - per_class_f1
output:
  save_predictions: false
  save_metrics: true
  output_dir: results/full_label_descriptions
"""

def sanitize_name(name: str) -> str:
    """Convert dataset name to safe filename."""
    return name.replace("/", "_").replace("-", "_")

def main():
    output_dir = Path("src/label_descriptions/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for dataset_name, (text_col, label_col, split) in DATASETS.items():
        dataset_short = sanitize_name(dataset_name)
        
        for model_key, (model_name, batch_size) in MODELS.items():
            for mode in MODES:
                filename = f"full_{dataset_short}_{model_key}_{mode}.yaml"
                filepath = output_dir / filename
                
                yaml_content = YAML_TEMPLATE.format(
                    dataset_short=dataset_short,
                    model=model_key,
                    mode=mode,
                    dataset_name=dataset_name,
                    split=split,
                    text_column=text_col,
                    label_column=label_col,
                    model_name=model_name,
                    batch_size=batch_size
                )
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(yaml_content)
                count += 1
                print(f"Created: {filename}")
    
    print(f"\nTotal: {count} experiment configs created in {output_dir}")

if __name__ == "__main__":
    main()
