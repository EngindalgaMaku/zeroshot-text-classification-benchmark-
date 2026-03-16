"""Run all 63 experiments WITHOUT modifying configs"""
import subprocess
from pathlib import Path
import shutil
import time

print("="*70)
print("CLEAN & RUN ALL 63 EXPERIMENTS")
print("="*70)

# Step 1: Clean old results
print("\n📦 Step 1: Cleaning old results...")
results_dir = Path("results/raw")
if results_dir.exists():
    # Backup old results
    backup_dir = Path(f"results/backup_{int(time.time())}")
    print(f"   Moving old results to: {backup_dir}")
    shutil.move(str(results_dir), str(backup_dir))
    print("   ✅ Old results backed up")

# Create fresh results directory
results_dir.mkdir(parents=True, exist_ok=True)
print("   ✅ Fresh results directory created\n")

# Step 2: Run experiments
print("🚀 Step 2: Running all experiments")
print("="*70)
print("Models: Qwen3-4B ✅, Jina-v2 ✅")
print("This will take 2-3 hours\n")

# 9 datasets × 7 models = 63 experiments
datasets = [
    "ag_news", "dbpedia_14", "yahoo_answers_topics", "banking77",
    "zeroshot_twitter_financial_news_sentiment", "SetFit_20_newsgroups",
    "go_emotions", "imdb", "sst2"
]

models = ["mpnet", "bge", "e5", "qwen3", "jina_v5", "instructor", "nomic"]

all_configs = []
for ds in datasets:
    for model in models:
        exp_name = f"exp_{ds}_{model}"
        config_path = f"experiments/{exp_name}.yaml"
        if Path(config_path).exists():
            all_configs.append((exp_name, config_path))

print(f"Total experiments: {len(all_configs)}\n")

results = []
for i, (exp_name, config_path) in enumerate(all_configs, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(all_configs)}] {exp_name}")
    print(f"{'='*70}")
    
    # Use conda run to ensure correct environment
    cmd = ["conda", "run", "-n", "zeroshot", "python", "main.py", "--config", config_path]
    
    try:
        subprocess.run(cmd, check=True, shell=True)
        results.append((exp_name, "✅"))
        print(f"✅ {exp_name} completed")
    except subprocess.CalledProcessError as e:
        results.append((exp_name, "❌"))
        print(f"❌ {exp_name} failed")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        break

# Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
success = sum(1 for _, status in results if status == "✅")
failed = sum(1 for _, status in results if status == "❌")
print(f"Completed: {len(results)}/{len(all_configs)}")
print(f"Success: {success}")
print(f"Failed: {failed}\n")

for exp, status in results:
    print(f"{status} {exp}")

print("\n" + "="*70)
print("Next steps:")
print("1. Check results in: results/raw/")
print("2. Run analysis: python scripts/generate_tables_and_plots.py")
print("="*70)