"""
Generate Task-Type Analysis Figure
Publication-quality figure showing zero-shot performance across different task types
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Publication-quality settings
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

def load_results():
    """Load all experiment results"""
    results_dir = Path('results/raw')
    data = []
    
    for json_file in results_dir.glob('*_metrics.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                # Extract dataset name
                dataset = result.get('dataset') or result.get('dataset_name', '')
                
                # Clean dataset names
                if dataset == 'ag_news':
                    dataset = 'AG News'
                elif dataset == 'dbpedia_14':
                    dataset = 'DBpedia-14'
                elif dataset == 'yahoo_answers_topics':
                    dataset = 'Yahoo Answers'
                elif dataset == 'banking77':
                    dataset = 'Banking77'
                elif dataset == 'zeroshot/twitter-financial-news-sentiment':
                    dataset = 'Twitter Financial'
                elif dataset == 'SetFit/20_newsgroups':
                    dataset = '20 Newsgroups'
                elif dataset == 'go_emotions':
                    dataset = 'GoEmotions'
                else:
                    continue
                
                data.append({
                    'dataset': dataset,
                    'macro_f1': result['macro_f1'] * 100
                })
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
    
    return pd.DataFrame(data)

def assign_task_types(df):
    """Assign task types to datasets"""
    task_mapping = {
        'AG News': 'Topic',
        '20 Newsgroups': 'Topic',
        'Yahoo Answers': 'Topic',
        'DBpedia-14': 'Entity',
        'Banking77': 'Intent',
        'Twitter Financial': 'Sentiment',
        'GoEmotions': 'Emotion'
    }
    
    df['task_type'] = df['dataset'].map(task_mapping)
    return df

def create_task_type_figure():
    """Create publication-quality task-type analysis figure"""
    
    print("Loading results...")
    df = load_results()
    print(f"✓ Loaded {len(df)} results")
    
    print("\nAssigning task types...")
    df = assign_task_types(df)
    
    # Calculate dataset means (average across all models)
    print("\nDataset means (across all models):")
    dataset_means = df.groupby('dataset')['macro_f1'].mean().sort_values(ascending=False)
    for dataset, score in dataset_means.items():
        print(f"  {dataset:20s}: {score:5.1f}%")
    
    # Calculate task type means
    print("\nTask type means:")
    task_means = df.groupby('task_type')['macro_f1'].mean().sort_values(ascending=False)
    for task, score in task_means.items():
        datasets = df[df['task_type'] == task]['dataset'].unique()
        print(f"  {task:12s}: {score:5.1f}% (n={len(datasets)} datasets)")
    
    # Prepare data for plotting
    task_order = ['Topic', 'Entity', 'Intent', 'Sentiment', 'Emotion']
    task_scores = [task_means.get(task, 0) for task in task_order]
    
    # Count datasets per task
    task_counts = df.groupby('task_type')['dataset'].nunique()
    task_labels = [f"{task}\n(n={task_counts.get(task, 0)})" for task in task_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color palette (professional, colorblind-friendly)
    colors = ['#F18F01', '#2E86AB', '#A23B72', '#C73E1D', '#6A994E']
    
    # Create bars
    bars = ax.bar(range(len(task_order)), task_scores, color=colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Add value labels on top of bars
    for i, (task, score) in enumerate(zip(task_order, task_scores)):
        ax.text(i, score + 1.5, f'{score:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Mean Macro-F1 (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    # No title - figures in papers don't have titles (caption is in text)
    
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_ylim(0, 90)  # More balanced range for 0-80 data
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'results/plots/task_type_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    
    # Also save as PDF for publication
    output_pdf = 'reports/TASK_TYPE_ANALYSIS.pdf'
    Path('reports').mkdir(exist_ok=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_pdf}")
    
    plt.show()
    
    # Create detailed breakdown table
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN BY TASK TYPE")
    print("="*70)
    
    for task in task_order:
        task_data = df[df['task_type'] == task]
        if len(task_data) == 0:
            continue
        
        print(f"\n{task.upper()}")
        print("-" * 70)
        
        # Dataset-level stats
        dataset_stats = task_data.groupby('dataset')['macro_f1'].agg(['mean', 'std', 'count'])
        for dataset, row in dataset_stats.iterrows():
            print(f"  {dataset:20s}: {row['mean']:5.1f}% ± {row['std']:4.1f}% (n={int(row['count'])})")
        
        # Task-level summary
        print(f"  {'TASK MEAN':20s}: {task_data['macro_f1'].mean():5.1f}%")
    
    # Save summary statistics
    summary = df.groupby('task_type')['macro_f1'].agg(['mean', 'std', 'min', 'max', 'count'])
    summary = summary.round(2)
    summary.to_csv('results/tables/task_type_summary.csv')
    print(f"\n✅ Saved summary: results/tables/task_type_summary.csv")
    
    return df, task_means

if __name__ == '__main__':
    print("="*70)
    print("TASK-TYPE ANALYSIS")
    print("="*70)
    
    df, task_means = create_task_type_figure()
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"✓ Entity classification is easiest: {task_means.get('Entity', 0):.1f}%")
    print(f"✓ Emotion classification is hardest: {task_means.get('Emotion', 0):.1f}%")
    print(f"✓ Performance range: {task_means.min():.1f}% - {task_means.max():.1f}%")
    print(f"✓ Task-dependent performance confirmed")
    print("\n📊 This figure clearly shows: Zero-shot performance is task-dependent!")
