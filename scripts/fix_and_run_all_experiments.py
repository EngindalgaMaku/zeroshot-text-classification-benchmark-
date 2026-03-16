"""
Fix all experiment configs with correct sample sizes and model names,
then run all 63 experiments (9 datasets × 7 models)
"""
import yaml
from pathlib import Path
import subprocess
import sys

# NOTEBOOK'TAN ALINAN DOĞRU BİLGİLER
datasets = [
    ("ag_news", "text", "label", "test", 1000),
    ("dbpedia_14", "content", "label", "test", 1000),
    ("yahoo_answers_topics", "best_answer", "topic", "test", 1000),
    ("banking77", "text", "label", "test", 1000),
    ("zeroshot/twitter-financial-news-sentiment", "text", "label", "validation", 1000),
    ("SetFit/20_newsgroups", "text", "label", "test", 2000),  # 2000!
    ("go_emotions", "text", "labels", "test", 1000),
    ("imdb", "text", "label", "test", 1000),
    ("sst2", "sentence", "label", "validation", 1000),
]

models = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet"),
    ("BAAI/bge-m3", "bge"),
    ("intfloat/multilingual-e5-large", "e5"),
    ("Qwen/Qwen3-Embedding-8B", "qwen3"),
    ("jinaai/jina-embeddings-v3", "jina_v3"),  # v3 kullanıyoruz
    ("hkunlp/instructor-large", "instructor"),
    ("nomic-ai/nomic-embed-text-v2-moe", "nomic"),
]

print("="*70)
print("STEP 1: Creating/Fixing all 63 experiment configs")
print("="*70)

Path("experiments").mkdir(exist_ok=True)
created = 0
updated = 0

for ds_name, text_col, label_col, split, max_samples in datasets:
    for model_name, model_short in models:
        ds_clean = ds_name.replace("/", "_").replace("-", "_")
        exp_name = f"exp_{ds_clean}_{model_short}"
        config_path = Path(f"experiments/{exp_name}.yaml")
        
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
                "batch_size": 32  # GPU ile
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
        
        if config_path.exists():
            updated += 1
            print(f"🔄 {exp_name}.yaml (updated)")
        else:
            created += 1
            print(f"✅ {exp_name}.yaml (created)")
            
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\nCreated: {created}, Updated: {updated}, Total: {created + updated}")
print(f"Expected: {len(datasets) * len(models)} (9 × 7 = 63)")

print("\n" + "="*70)
print("STEP 2: Running all experiments")
print("="*70)
print("This will take 2-3 hours with GPU")
print("Press Ctrl+C to cancel\n")

import time
time.sleep(3)

# Tüm config dosyalarını topla
all_configs = []
for ds_name, _, _, _, _ in datasets:
    for _, model_short in models:
        ds_clean = ds_name.replace("/", "_").replace("-", "_")
        exp_name = f"exp_{ds_clean}_{model_short}"
        config_path = f"experiments/{exp_name}.yaml"
        all_configs.append((exp_name, config_path))

print(f"Total experiments to run: {len(all_configs)}\n")

results = []
for i, (exp_name, config_path) in enumerate(all_configs, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(all_configs)}] {exp_name}")
    print(f"{'='*70}")
    
    # main.py'ı çalıştır
    cmd = ["python", "main.py", "--config", config_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        results.append((exp_name, "✅"))
        print(f"✅ {exp_name} completed")
    except subprocess.CalledProcessError as e:
        results.append((exp_name, "❌"))
        print(f"❌ {exp_name} failed: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        break

# Sonuçları göster
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
success = sum(1 for _, status in results if status == "✅")
failed = sum(1 for _, status in results if status == "❌")
print(f"Completed: {i}/{len(all_configs)}")
print(f"Success: {success}")
print(f"Failed: {failed}")
print("\nDetailed results:")
for exp, status in results:
    print(f"{status} {exp}")

print("\n" + "="*70)
print("Next: Run analysis scripts to generate tables and plots")
print("="*70)