"""Analyze error patterns and identify most confused class pairs.

This script identifies the top 5 most confused class pairs for each dataset
and provides detailed analysis for GoEmotions and Yahoo Answers.

**Validates: Requirements 10.2, 10.3, 10.4**
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from labels import LABEL_SETS

# Datasets to analyze
DATASETS = ["ag_news", "banking77", "go_emotions", "yahoo_answers_topics", "imdb", "sst2"]

# Model names mapping
MODEL_NAMES = {
    "bge": "BGE-M3",
    "e5": "E5-large",
    "instructor": "Instructor",
    "jina_v5": "Jina-v5",
    "mpnet": "MPNet",
    "nomic": "Nomic",
    "qwen3": "Qwen3"
}

# Dataset display names
DATASET_NAMES = {
    "ag_news": "AG News",
    "banking77": "Banking77",
    "go_emotions": "GoEmotions",
    "yahoo_answers_topics": "Yahoo Answers",
    "imdb": "IMDB",
    "sst2": "SST-2",
}


def load_predictions(dataset, model):
    """Load prediction CSV file for a dataset-model combination."""
    pred_file = Path(f"results/raw/{dataset}_{model}_predictions.csv")
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def get_label_names(dataset):
    """Get label names for a dataset."""
    # Map dataset names to label set keys
    label_key_map = {
        "ag_news": "ag_news",
        "banking77": "banking77",
        "go_emotions": "go_emotions",
        "yahoo_answers_topics": "yahoo_answers_topics",
        "imdb": "imdb",
        "sst2": "sst2",
    }
    
    label_key = label_key_map.get(dataset)
    if label_key and label_key in LABEL_SETS:
        labels = LABEL_SETS[label_key]["name_only"]
        # Extract just the first label text for each class
        return [labels[i][0] for i in sorted(labels.keys())]
    return None


def find_top_confused_pairs(y_true, y_pred, labels, top_n=5):
    """Find the top N most confused class pairs (excluding diagonal).
    
    Returns list of tuples: (true_label, pred_label, count, percentage)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Get off-diagonal elements (errors only)
    confused_pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:  # Exclude correct predictions
                count = cm[i, j]
                if count > 0:
                    # Calculate percentage of true class i that was predicted as j
                    total_true_i = cm[i, :].sum()
                    percentage = (count / total_true_i * 100) if total_true_i > 0 else 0
                    confused_pairs.append((labels[i], labels[j], count, percentage))
    
    # Sort by count (descending)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return confused_pairs[:top_n]


def analyze_dataset_errors(dataset, models):
    """Analyze error patterns for a specific dataset across all models."""
    print(f"\n{'='*80}")
    print(f"Dataset: {DATASET_NAMES[dataset]}")
    print(f"{'='*80}")
    
    labels = get_label_names(dataset)
    if labels is None:
        print(f"⚠️  Warning: No labels found for {dataset}")
        return None
    
    print(f"Classes: {len(labels)}")
    
    # Aggregate confusion across all models
    all_confused_pairs = defaultdict(lambda: {"count": 0, "models": []})
    
    results = []
    
    for model_key, model_name in models.items():
        df = load_predictions(dataset, model_key)
        
        if df is None:
            print(f"⚠️  {model_name}: Prediction file not found")
            continue
        
        # Find top confused pairs for this model
        top_pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values, labels, top_n=5)
        
        print(f"\n{model_name} - Top 5 Confused Pairs:")
        for i, (true_label, pred_label, count, pct) in enumerate(top_pairs, 1):
            print(f"  {i}. '{true_label}' → '{pred_label}': {count} errors ({pct:.1f}%)")
            
            # Aggregate across models
            pair_key = (true_label, pred_label)
            all_confused_pairs[pair_key]["count"] += count
            all_confused_pairs[pair_key]["models"].append(model_name)
            
            results.append({
                "dataset": dataset,
                "model": model_name,
                "rank": i,
                "true_label": true_label,
                "predicted_label": pred_label,
                "error_count": count,
                "percentage": pct
            })
    
    # Show aggregate top confused pairs across all models
    print(f"\n{DATASET_NAMES[dataset]} - Aggregate Top 10 Confused Pairs (All Models):")
    sorted_pairs = sorted(all_confused_pairs.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    
    for i, ((true_label, pred_label), data) in enumerate(sorted_pairs, 1):
        model_count = len(data["models"])
        print(f"  {i}. '{true_label}' → '{pred_label}': {data['count']} total errors ({model_count} models)")
    
    return results, sorted_pairs


def analyze_goemotions_patterns(models):
    """Detailed analysis of GoEmotions fine-grained emotion confusions.
    
    **Validates: Requirement 10.3**
    """
    print(f"\n{'='*80}")
    print("GOEMOTIONS FINE-GRAINED EMOTION ANALYSIS")
    print(f"{'='*80}")
    
    dataset = "go_emotions"
    labels = get_label_names(dataset)
    
    # Define emotion groups for analysis
    emotion_groups = {
        "positive": ["admiration", "amusement", "approval", "caring", "excitement", 
                    "gratitude", "joy", "love", "optimism", "pride", "relief"],
        "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust",
                    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"],
        "ambiguous": ["confusion", "curiosity", "desire", "realization", "surprise"],
        "neutral": ["neutral"]
    }
    
    # Reverse mapping: emotion -> group
    emotion_to_group = {}
    for group, emotions in emotion_groups.items():
        for emotion in emotions:
            emotion_to_group[emotion] = group
    
    # Analyze cross-group confusions
    cross_group_confusions = defaultdict(int)
    within_group_confusions = defaultdict(int)
    
    for model_key, model_name in models.items():
        df = load_predictions(dataset, model_key)
        if df is None:
            continue
        
        cm = confusion_matrix(df['y_true'].values, df['y_pred'].values)
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    true_emotion = labels[i]
                    pred_emotion = labels[j]
                    
                    true_group = emotion_to_group.get(true_emotion, "unknown")
                    pred_group = emotion_to_group.get(pred_emotion, "unknown")
                    
                    if true_group == pred_group:
                        within_group_confusions[true_group] += cm[i, j]
                    else:
                        pair = tuple(sorted([true_group, pred_group]))
                        cross_group_confusions[pair] += cm[i, j]
    
    print("\nWithin-Group Confusions (same valence):")
    for group, count in sorted(within_group_confusions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count} errors")
    
    print("\nCross-Group Confusions (different valence):")
    for pair, count in sorted(cross_group_confusions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pair[0]} ↔ {pair[1]}: {count} errors")
    
    # Find most confused fine-grained emotions
    print("\nMost Problematic Fine-Grained Emotion Distinctions:")
    all_pairs = []
    for model_key in models.keys():
        df = load_predictions(dataset, model_key)
        if df is None:
            continue
        pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values, labels, top_n=10)
        all_pairs.extend(pairs)
    
    # Aggregate
    pair_counts = defaultdict(int)
    for true_label, pred_label, count, _ in all_pairs:
        pair_key = (true_label, pred_label)
        pair_counts[pair_key] += count
    
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, ((true_label, pred_label), count) in enumerate(sorted_pairs, 1):
        true_group = emotion_to_group.get(true_label, "unknown")
        pred_group = emotion_to_group.get(pred_label, "unknown")
        valence = "same valence" if true_group == pred_group else "different valence"
        print(f"  {i}. '{true_label}' → '{pred_label}': {count} errors ({valence})")


def analyze_yahoo_patterns(models):
    """Detailed analysis of Yahoo Answers broad category confusions.
    
    **Validates: Requirement 10.4**
    """
    print(f"\n{'='*80}")
    print("YAHOO ANSWERS BROAD CATEGORY ANALYSIS")
    print(f"{'='*80}")
    
    dataset = "yahoo_answers_topics"
    labels = get_label_names(dataset)
    
    if labels is None:
        print("⚠️  Warning: No labels found for Yahoo Answers")
        return
    
    # Define category relationships
    category_groups = {
        "knowledge": ["science and mathematics", "education and reference", "health"],
        "technology": ["computers and internet"],
        "social": ["society and culture", "family and relationships", "politics and government"],
        "leisure": ["sports", "entertainment and music"],
        "practical": ["business and finance"]
    }
    
    # Reverse mapping
    category_to_group = {}
    for group, categories in category_groups.items():
        for category in categories:
            category_to_group[category] = group
    
    # Analyze confusions between broad category groups
    cross_group_confusions = defaultdict(int)
    within_group_confusions = defaultdict(int)
    
    for model_key, model_name in models.items():
        df = load_predictions(dataset, model_key)
        if df is None:
            continue
        
        cm = confusion_matrix(df['y_true'].values, df['y_pred'].values)
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    true_cat = labels[i]
                    pred_cat = labels[j]
                    
                    true_group = category_to_group.get(true_cat, "other")
                    pred_group = category_to_group.get(pred_cat, "other")
                    
                    if true_group == pred_group:
                        within_group_confusions[true_group] += cm[i, j]
                    else:
                        pair = tuple(sorted([true_group, pred_group]))
                        cross_group_confusions[pair] += cm[i, j]
    
    print("\nWithin-Group Confusions (related topics):")
    for group, count in sorted(within_group_confusions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count} errors")
    
    print("\nCross-Group Confusions (unrelated topics):")
    for pair, count in sorted(cross_group_confusions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair[0]} ↔ {pair[1]}: {count} errors")
    
    # Find most confused category pairs
    print("\nMost Confused Category Pairs:")
    all_pairs = []
    for model_key in models.keys():
        df = load_predictions(dataset, model_key)
        if df is None:
            continue
        pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values, labels, top_n=10)
        all_pairs.extend(pairs)
    
    # Aggregate
    pair_counts = defaultdict(int)
    for true_label, pred_label, count, _ in all_pairs:
        pair_key = (true_label, pred_label)
        pair_counts[pair_key] += count
    
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, ((true_label, pred_label), count) in enumerate(sorted_pairs, 1):
        true_group = category_to_group.get(true_label, "other")
        pred_group = category_to_group.get(pred_label, "other")
        relation = "related" if true_group == pred_group else "unrelated"
        print(f"  {i}. '{true_label}' → '{pred_label}': {count} errors ({relation})")


def main():
    """Main error pattern analysis."""
    print("=" * 80)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Analyze each dataset
    for dataset in DATASETS:
        results, aggregate_pairs = analyze_dataset_errors(dataset, MODEL_NAMES)
        if results:
            all_results.extend(results)
    
    # Save detailed results to CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_file = output_dir / "error_patterns_detailed.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ Saved detailed error patterns to: {output_file}")
    
    # Specialized analyses
    analyze_goemotions_patterns(MODEL_NAMES)
    analyze_yahoo_patterns(MODEL_NAMES)
    
    print("\n" + "=" * 80)
    print("✅ ERROR PATTERN ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
