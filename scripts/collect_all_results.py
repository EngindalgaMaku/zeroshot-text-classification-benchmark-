"""Collect all experiment results from JSON files"""
import json
import pandas as pd
import glob
from pathlib import Path

# Change to results/raw directory
results_dir = Path("results/raw")

rows = []
for f in results_dir.glob("*.json"):
    try:
        with open(f, encoding='utf-8') as fp:
            d = json.load(fp)
        
        # Extract model name from various fields
        model = None
        if "biencoder" in d:
            model = d["biencoder"]
        elif "reranker" in d:
            model = d["reranker"]
        elif "model" in d:
            model = d["model"]
        else:
            # Try to extract from experiment name
            exp_name = d.get("experiment_name", f.stem)
            if "instructor" in exp_name:
                model = "INSTRUCTOR"
            elif "qwen" in exp_name:
                model = "Qwen3"
            elif "snowflake" in exp_name or "arctic" in exp_name:
                model = "Snowflake"
            elif "jina" in exp_name:
                model = "Jina-v5"
            elif "bge" in exp_name:
                model = "BGE-M3"
            elif "e5" in exp_name:
                model = "E5-large"
            elif "mpnet" in exp_name:
                model = "MPNet"
            else:
                model = "Unknown"
        
        # Extract dataset name
        dataset = d.get("dataset") or d.get("dataset_name", "Unknown")
        
        # Clean up dataset names
        if dataset == "ag_news":
            dataset = "AG News"
        elif dataset == "dbpedia_14":
            dataset = "DBpedia-14"
        elif dataset == "yahoo_answers_topics":
            dataset = "Yahoo Answers"
        elif dataset == "banking77":
            dataset = "Banking77"
        elif dataset == "zeroshot/twitter-financial-news-sentiment":
            dataset = "Twitter Financial"
        elif dataset == "SetFit/20_newsgroups":
            dataset = "20 Newsgroups"
        elif dataset == "go_emotions":
            dataset = "GoEmotions"
        
        rows.append({
            "dataset": dataset,
            "model": model,
            "macro_f1": round(d["macro_f1"] * 100, 2),  # Convert to percentage
            "accuracy": round(d["accuracy"] * 100, 2),
            "weighted_f1": round(d.get("weighted_f1", 0) * 100, 2),
            "samples": d.get("num_samples") or d.get("total_samples", 0),
            "experiment": d.get("experiment_name", f.stem)
        })
    except Exception as e:
        print(f"⚠️  Error reading {f.name}: {e}")

df = pd.DataFrame(rows)

# Sort by dataset and model
df = df.sort_values(["dataset", "model"])

print("\n" + "="*80)
print("ALL EXPERIMENT RESULTS")
print("="*80)
print(f"\nTotal experiments: {len(df)}")
print(f"Datasets: {df['dataset'].nunique()}")
print(f"Models: {df['model'].nunique()}")
print("\n" + df.to_string(index=False))

# Save to CSV
output_file = "results/ALL_RESULTS.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY BY MODEL")
print("="*80)
model_summary = df.groupby("model")[["macro_f1", "accuracy"]].agg(["mean", "std", "min", "max"])
print(model_summary.round(2))

print("\n" + "="*80)
print("SUMMARY BY DATASET")
print("="*80)
dataset_summary = df.groupby("dataset")[["macro_f1", "accuracy"]].agg(["mean", "std", "min", "max"])
print(dataset_summary.round(2))

# Pivot table for heatmap
print("\n" + "="*80)
print("MACRO F1 SCORES (%) - PIVOT TABLE")
print("="*80)

# Check for duplicates
duplicates = df[df.duplicated(subset=["model", "dataset"], keep=False)]
if len(duplicates) > 0:
    print("\n⚠️  WARNING: Found duplicate model+dataset combinations:")
    print(duplicates[["dataset", "model", "macro_f1", "experiment"]].sort_values(["dataset", "model"]))
    print("\n📋 Keeping the best F1 score for each model+dataset pair...")
    
    # Keep only the best F1 score for each model+dataset
    df = df.sort_values("macro_f1", ascending=False).drop_duplicates(subset=["model", "dataset"], keep="first")
    print(f"✅ Reduced to {len(df)} unique experiments\n")

pivot = df.pivot(index="model", columns="dataset", values="macro_f1")
print(pivot.round(1))

# Save pivot
pivot.to_csv("results/PIVOT_F1_SCORES.csv")
print(f"\n✅ Saved pivot table to: results/PIVOT_F1_SCORES.csv")
