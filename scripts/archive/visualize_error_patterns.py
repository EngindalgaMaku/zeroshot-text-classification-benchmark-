"""Visualize error patterns across datasets and models.

This script generates publication-quality visualizations of error patterns
including bar charts of most confused class pairs and cross-model comparisons.

**Validates: Requirement 10.5**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from labels import LABEL_SETS

# Publication-quality styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Datasets to analyze
DATASETS = ["ag_news", "banking77", "go_emotions", "yahoo_answers_topics"]

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
    "yahoo_answers_topics": "Yahoo Answers"
}


def load_predictions(dataset, model):
    """Load prediction CSV file for a dataset-model combination."""
    pred_file = Path(f"results/raw/{dataset}_{model}_predictions.csv")
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def get_label_names(dataset):
    """Get label names for a dataset."""
    label_key_map = {
        "ag_news": "ag_news",
        "banking77": "banking77",
        "go_emotions": "go_emotions",
        "yahoo_answers_topics": "yahoo_answers_topics"
    }
    
    label_key = label_key_map.get(dataset)
    if label_key and label_key in LABEL_SETS:
        labels = LABEL_SETS[label_key]["name_only"]
        return [labels[i][0] for i in sorted(labels.keys())]
    return None


def find_top_confused_pairs(y_true, y_pred, labels, top_n=5):
    """Find the top N most confused class pairs."""
    cm = confusion_matrix(y_true, y_pred)
    confused_pairs = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    total_true_i = cm[i, :].sum()
                    percentage = (count / total_true_i * 100) if total_true_i > 0 else 0
                    confused_pairs.append((labels[i], labels[j], count, percentage))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs[:top_n]


def plot_top_confused_pairs_by_dataset(output_dir):
    """Generate bar charts showing most confused class pairs for each dataset."""
    print("\n📊 Generating top confused pairs visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top 10 Most Confused Class Pairs by Dataset (All Models Combined)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx // 2, idx % 2]
        labels = get_label_names(dataset)
        
        if labels is None:
            ax.text(0.5, 0.5, f'No data for {dataset}', ha='center', va='center')
            continue
        
        # Aggregate confusion across all models
        all_confused_pairs = defaultdict(int)
        
        for model_key in MODEL_NAMES.keys():
            df = load_predictions(dataset, model_key)
            if df is None:
                continue
            
            top_pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values, 
                                               labels, top_n=20)
            for true_label, pred_label, count, _ in top_pairs:
                pair_key = (true_label, pred_label)
                all_confused_pairs[pair_key] += count
        
        # Get top 10
        sorted_pairs = sorted(all_confused_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_pairs:
            pair_labels = [f"{true_l[:20]}→{pred_l[:20]}" for (true_l, pred_l), _ in sorted_pairs]
            counts = [count for _, count in sorted_pairs]
            
            bars = ax.barh(range(len(pair_labels)), counts, color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(pair_labels)))
            ax.set_yticklabels(pair_labels, fontsize=9)
            ax.set_xlabel('Total Error Count', fontweight='bold')
            ax.set_title(f'{DATASET_NAMES[dataset]} ({len(labels)} classes)', 
                        fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(counts):
                ax.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center')
    
    plt.tight_layout()
    output_file = output_dir / "error_patterns_top_confused_pairs.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    eps_file = output_dir / "error_patterns_top_confused_pairs.eps"
    plt.savefig(eps_file, dpi=300, bbox_inches='tight', format='eps')
    plt.close()
    print(f"   ✅ {output_file.name}")
    print(f"   ✅ {eps_file.name}")


def plot_model_comparison_error_patterns(output_dir):
    """Generate visualizations comparing error patterns across models."""
    print("\n📊 Generating model comparison visualizations...")
    
    # For each dataset, compare models
    for dataset in DATASETS:
        labels = get_label_names(dataset)
        if labels is None:
            continue
        
        # Collect top confused pair for each model
        model_top_pairs = {}
        
        for model_key, model_name in MODEL_NAMES.items():
            df = load_predictions(dataset, model_key)
            if df is None:
                continue
            
            top_pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values,
                                               labels, top_n=5)
            if top_pairs:
                model_top_pairs[model_name] = top_pairs
        
        if not model_top_pairs:
            continue
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        models = list(model_top_pairs.keys())
        x = np.arange(len(models))
        width = 0.15
        
        # Get up to 5 most common pairs across all models
        all_pairs = defaultdict(int)
        for pairs in model_top_pairs.values():
            for true_l, pred_l, count, _ in pairs:
                all_pairs[(true_l, pred_l)] += count
        
        common_pairs = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot bars for each common pair
        colors = plt.cm.Set3(np.linspace(0, 1, len(common_pairs)))
        
        for i, ((true_l, pred_l), _) in enumerate(common_pairs):
            counts = []
            for model in models:
                # Find this pair in model's top pairs
                pair_count = 0
                for t, p, c, _ in model_top_pairs[model]:
                    if t == true_l and p == pred_l:
                        pair_count = c
                        break
                counts.append(pair_count)
            
            ax.bar(x + i * width, counts, width, label=f'{true_l[:15]}→{pred_l[:15]}',
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Error Count', fontweight='bold', fontsize=12)
        ax.set_title(f'{DATASET_NAMES[dataset]} - Error Pattern Comparison Across Models',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(title='Confused Pairs', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"error_patterns_{dataset}_model_comparison.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"   ✅ {output_file.name}")


def plot_goemotions_emotion_groups(output_dir):
    """Visualize GoEmotions error patterns by emotion groups."""
    print("\n📊 Generating GoEmotions emotion group analysis...")
    
    dataset = "go_emotions"
    labels = get_label_names(dataset)
    
    # Define emotion groups
    emotion_groups = {
        "positive": ["admiration", "amusement", "approval", "caring", "excitement",
                    "gratitude", "joy", "love", "optimism", "pride", "relief"],
        "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust",
                    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"],
        "ambiguous": ["confusion", "curiosity", "desire", "realization", "surprise"],
        "neutral": ["neutral"]
    }
    
    emotion_to_group = {}
    for group, emotions in emotion_groups.items():
        for emotion in emotions:
            emotion_to_group[emotion] = group
    
    # Analyze confusions
    within_group = defaultdict(int)
    cross_group = defaultdict(int)
    
    for model_key in MODEL_NAMES.keys():
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
                        within_group[true_group] += cm[i, j]
                    else:
                        pair = tuple(sorted([true_group, pred_group]))
                        cross_group[pair] += cm[i, j]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('GoEmotions Error Patterns by Emotion Valence Groups',
                 fontsize=14, fontweight='bold')
    
    # Within-group confusions
    groups = list(within_group.keys())
    counts = list(within_group.values())
    ax1.barh(groups, counts, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Error Count', fontweight='bold')
    ax1.set_title('Within-Group Confusions\n(Same Valence)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(counts):
        ax1.text(v + max(counts)*0.01, i, str(v), va='center')
    
    # Cross-group confusions
    sorted_cross = sorted(cross_group.items(), key=lambda x: x[1], reverse=True)[:8]
    pair_labels = [f"{p[0]}↔{p[1]}" for p, _ in sorted_cross]
    pair_counts = [c for _, c in sorted_cross]
    
    ax2.barh(range(len(pair_labels)), pair_counts, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(pair_labels)))
    ax2.set_yticklabels(pair_labels)
    ax2.set_xlabel('Error Count', fontweight='bold')
    ax2.set_title('Cross-Group Confusions\n(Different Valence)', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(pair_counts):
        ax2.text(v + max(pair_counts)*0.01, i, str(v), va='center')
    
    plt.tight_layout()
    output_file = output_dir / "error_patterns_goemotions_valence_groups.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"   ✅ {output_file.name}")


def plot_yahoo_category_groups(output_dir):
    """Visualize Yahoo Answers error patterns by category groups."""
    print("\n📊 Generating Yahoo Answers category group analysis...")
    
    dataset = "yahoo_answers_topics"
    labels = get_label_names(dataset)
    
    if labels is None:
        print("   ⚠️  No labels found for Yahoo Answers")
        return
    
    # Define category groups
    category_groups = {
        "knowledge": ["science and mathematics", "education and reference", "health"],
        "technology": ["computers and internet"],
        "social": ["society and culture", "family and relationships", "politics and government"],
        "leisure": ["sports", "entertainment and music"],
        "practical": ["business and finance"]
    }
    
    category_to_group = {}
    for group, categories in category_groups.items():
        for category in categories:
            category_to_group[category] = group
    
    # Analyze confusions
    within_group = defaultdict(int)
    cross_group = defaultdict(int)
    
    for model_key in MODEL_NAMES.keys():
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
                        within_group[true_group] += cm[i, j]
                    else:
                        pair = tuple(sorted([true_group, pred_group]))
                        cross_group[pair] += cm[i, j]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Yahoo Answers Error Patterns by Topic Groups',
                 fontsize=14, fontweight='bold')
    
    # Within-group confusions
    groups = list(within_group.keys())
    counts = list(within_group.values())
    ax1.barh(groups, counts, color='lightgreen', alpha=0.7)
    ax1.set_xlabel('Error Count', fontweight='bold')
    ax1.set_title('Within-Group Confusions\n(Related Topics)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(counts):
        ax1.text(v + max(counts)*0.01, i, str(v), va='center')
    
    # Cross-group confusions
    sorted_cross = sorted(cross_group.items(), key=lambda x: x[1], reverse=True)[:8]
    pair_labels = [f"{p[0]}↔{p[1]}" for p, _ in sorted_cross]
    pair_counts = [c for _, c in sorted_cross]
    
    ax2.barh(range(len(pair_labels)), pair_counts, color='orange', alpha=0.7)
    ax2.set_yticks(range(len(pair_labels)))
    ax2.set_yticklabels(pair_labels)
    ax2.set_xlabel('Error Count', fontweight='bold')
    ax2.set_title('Cross-Group Confusions\n(Unrelated Topics)', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(pair_counts):
        ax2.text(v + max(pair_counts)*0.01, i, str(v), va='center')
    
    plt.tight_layout()
    output_file = output_dir / "error_patterns_yahoo_topic_groups.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"   ✅ {output_file.name}")


def create_summary_error_figure():
    """Create a single publication figure showing top-5 confused pairs for 3 key datasets."""
    print("\n📊 Generating summary error pattern figure...")

    summary_datasets = ["ag_news", "banking77", "go_emotions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Top-5 Most Confused Class Pairs (All Models Combined)',
                 fontsize=14, fontweight='bold')

    colors = ['#4878CF', '#6ACC65', '#D65F5F']

    for idx, dataset in enumerate(summary_datasets):
        ax = axes[idx]
        labels = get_label_names(dataset)

        if labels is None:
            ax.text(0.5, 0.5, f'No data for {dataset}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(DATASET_NAMES.get(dataset, dataset))
            continue

        # Aggregate confusion across all models
        all_confused_pairs = defaultdict(int)
        for model_key in MODEL_NAMES.keys():
            df = load_predictions(dataset, model_key)
            if df is None:
                continue
            top_pairs = find_top_confused_pairs(df['y_true'].values, df['y_pred'].values,
                                                labels, top_n=20)
            for true_label, pred_label, count, _ in top_pairs:
                all_confused_pairs[(true_label, pred_label)] += count

        sorted_pairs = sorted(all_confused_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

        if sorted_pairs:
            pair_labels = [f"{true_l[:18]}→{pred_l[:18]}" for (true_l, pred_l), _ in sorted_pairs]
            counts = [count for _, count in sorted_pairs]

            # Horizontal bar chart, top pair at top
            y_pos = range(len(pair_labels) - 1, -1, -1)
            ax.barh(list(y_pos), counts, color=colors[idx], alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(pair_labels, fontsize=9)
            ax.set_xlabel('Total Error Count')
            ax.set_title(DATASET_NAMES.get(dataset, dataset))
            ax.grid(axis='x', alpha=0.3)

            max_count = max(counts)
            for i, (y, v) in enumerate(zip(y_pos, counts)):
                ax.text(v + max_count * 0.02, y, str(v), va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(DATASET_NAMES.get(dataset, dataset))

    plt.tight_layout()

    output_base = Path("results/plots/error_patterns_summary")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in ('pdf', 'eps', 'png'):
        out = output_base.with_suffix(f'.{fmt}')
        plt.savefig(out, format=fmt)
        print(f"   ✅ {out.name}")

    plt.close()


def main():
    """Generate all error pattern visualizations."""
    print("=" * 80)
    print("GENERATING ERROR PATTERN VISUALIZATIONS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("results/plots/error_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Generate visualizations
    plot_top_confused_pairs_by_dataset(output_dir)
    plot_model_comparison_error_patterns(output_dir)
    plot_goemotions_emotion_groups(output_dir)
    plot_yahoo_category_groups(output_dir)
    create_summary_error_figure()
    
    print("\n" + "=" * 80)
    print("✅ ERROR PATTERN VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All visualizations saved to: {output_dir}")
    print("   Summary figure: results/plots/error_patterns_summary.{pdf,eps,png}")
    print("   Format: PDF and EPS (publication-quality)")


if __name__ == "__main__":
    main()
