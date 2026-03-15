"""
Generate F1 Score Heatmap Report (PDF)
Publication-quality figure for academic papers
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Publication-quality settings (IEEE/Nature/Elsevier standard)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

def load_results():
    """Load all experiment results"""
    results_dir = Path('results/raw')
    results = []
    
    for json_file in results_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"  ⚠ Could not load {json_file.name}: {e}")
    
    return results

def extract_model_dataset_scores(results):
    """Extract F1 scores in model x dataset format"""
    # Model name mapping (academic literature uses "embedding model")
    model_map = {
        'bge': 'BGE-M3',
        'e5': 'E5-large',
        'gte': 'GTE-large',
        'instructor': 'INSTRUCTOR',
        'jina': 'Jina-v3',
        'mpnet': 'MPNet',
        'qwen': 'Qwen3-4B',
        'snowflake': 'Snowflake'
    }
    
    # Dataset name mapping (shorter names for readability)
    dataset_map = {
        'ag_news': 'AG News',
        'dbpedia': 'DBPedia',
        'yahoo': 'Yahoo',
        'banking77': 'Banking77',
        '20_newsgroups': '20News',
        'setfit/20_newsgroups': '20News',
        'twitter_financial': 'Twitter-Fin',
        'zeroshot_twitter': 'Twitter-Fin',
        'go_emotions': 'GoEmotions',
        'goemotions': 'GoEmotions'
    }
    
    # Initialize score matrix
    data_dict = {}
    
    print("\n  Detected experiments:")
    for result in results:
        exp_name = result.get('experiment_name', '').lower()
        # Try both 'dataset_name' and 'dataset' fields
        dataset_name = result.get('dataset_name', result.get('dataset', '')).lower()
        macro_f1 = result.get('macro_f1', 0) * 100  # Convert to percentage
        
        # Identify model
        model = None
        for key, name in model_map.items():
            if key in exp_name:
                model = name
                break
        
        # Identify dataset
        dataset = None
        for key, name in dataset_map.items():
            if key in dataset_name or key in exp_name:
                dataset = name
                break
        
        if model and dataset:
            if model not in data_dict:
                data_dict[model] = {}
            # Take the maximum F1 if duplicate (shouldn't happen)
            if dataset in data_dict[model]:
                data_dict[model][dataset] = max(data_dict[model][dataset], macro_f1)
            else:
                data_dict[model][dataset] = macro_f1
            print(f"    ✓ {model:12s} × {dataset:12s} = {macro_f1:.1f}%")
        else:
            if not model:
                print(f"    ✗ Unknown model in: {exp_name[:40]}")
            if not dataset:
                print(f"    ✗ Unknown dataset in: {dataset_name[:40]}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict).T
    
    # Ensure consistent column order
    column_order = ['20News', 'AG News', 'Banking77', 'DBPedia', 'GoEmotions', 'Twitter-Fin', 'Yahoo']
    df = df[[col for col in column_order if col in df.columns]]
    
    # Ensure consistent row order (INSTRUCTOR + Snowflake added!)
    row_order = ['INSTRUCTOR', 'Snowflake', 'Qwen3-4B', 'E5-large', 'MPNet', 'Jina-v3', 'BGE-M3']
    df = df.reindex([row for row in row_order if row in df.index])
    
    return df

def create_publication_heatmap(fig, df):
    """
    Create publication-quality heatmap
    Following Nature/IEEE/ACL guidelines
    """
    ax = fig.add_subplot(111)
    
    # Viridis is standard in academic publications
    cmap = 'viridis'
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=40, 
                vmax=85,
                cbar_kws={
                    'label': 'Macro F1 (%)',
                    'shrink': 0.85,
                    'aspect': 20,
                    'pad': 0.02
                },
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                square=False,
                annot_kws={'size': 9, 'weight': 'normal'}
    )
    
    # Labels (NO TITLE - goes in manuscript caption)
    # Academic terminology: "Embedding Model" not just "Model"
    ax.set_xlabel('Dataset', fontsize=11, weight='normal')
    ax.set_ylabel('Embedding Model', fontsize=11, weight='normal')
    
    # Fix overlapping labels
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=45, 
                       ha='right',
                       rotation_mode='anchor',
                       fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), 
                       rotation=0,
                       fontsize=10)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()

def create_grayscale_heatmap(fig, df):
    """Grayscale version for print journals"""
    ax = fig.add_subplot(111)
    
    cmap = 'gray_r'
    
    sns.heatmap(df, 
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=40, 
                vmax=85,
                cbar_kws={
                    'label': 'Macro F1 (%)',
                    'shrink': 0.85,
                    'aspect': 20,
                    'pad': 0.02
                },
                linewidths=0.5,
                linecolor='black',
                ax=ax,
                square=False,
                annot_kws={'size': 9, 'weight': 'normal'}
    )
    
    ax.set_xlabel('Dataset', fontsize=11, weight='normal')
    ax.set_ylabel('Embedding Model', fontsize=11, weight='normal')
    
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=45, 
                       ha='right',
                       rotation_mode='anchor',
                       fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), 
                       rotation=0,
                       fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()

def create_statistics_table(fig, df):
    """Statistics table for supplementary material"""
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Model statistics
    model_stats = []
    for model in df.index:
        scores = df.loc[model].values
        model_stats.append([
            model,
            f"{scores.mean():.1f}",
            f"{scores.std():.2f}",
            f"{scores.min():.1f}",
            f"{scores.max():.1f}"
        ])
    
    headers = ['Embedding Model', 'Mean', 'Std', 'Min', 'Max']
    
    ax_table = fig.add_axes([0.15, 0.55, 0.7, 0.35])
    ax_table.axis('off')
    
    table = ax_table.table(
        cellText=model_stats,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    
    for i in range(1, len(model_stats) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    caption = ("Table 1. Performance statistics of sentence embedding models in zero-shot "
               "text classification across seven benchmark datasets.")
    fig.text(0.5, 0.45, caption, ha='center', va='top', fontsize=9, 
             style='italic', wrap=True)
    
    # Dataset statistics
    dataset_stats = []
    for dataset in df.columns:
        scores = df[dataset].values
        dataset_stats.append([
            dataset,
            f"{scores.mean():.1f}",
            f"{scores.std():.2f}",
            f"{scores.min():.1f}",
            f"{scores.max():.1f}"
        ])
    
    headers2 = ['Dataset', 'Mean', 'Std', 'Min', 'Max']
    
    ax_table2 = fig.add_axes([0.15, 0.1, 0.7, 0.30])
    ax_table2.axis('off')
    
    table2 = ax_table2.table(
        cellText=dataset_stats,
        colLabels=headers2,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.8)
    
    for i in range(len(headers2)):
        cell = table2[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    
    for i in range(1, len(dataset_stats) + 1):
        for j in range(len(headers2)):
            cell = table2[(i, j)]
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    caption2 = "Table 2. Dataset difficulty analysis based on mean F1 scores across all models."
    fig.text(0.5, 0.02, caption2, ha='center', va='bottom', fontsize=9,
             style='italic', wrap=True)

def generate_pdf():
    """Generate publication-ready PDF"""
    from pathlib import Path
    
    # Ensure reports directory exists
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = 'reports/F1_HEATMAP_PUBLICATION.pdf'
    
    print("Loading experiment results...")
    results = load_results()
    print(f"  ✓ Loaded {len(results)} valid JSON files")
    
    print("\nExtracting F1 scores...")
    df = extract_model_dataset_scores(results)
    print(f"\n  ✓ Final matrix: {df.shape[0]} models × {df.shape[1]} datasets")
    
    print("\nGenerating publication-quality PDF...")
    
    with PdfPages(output_file) as pdf:
        # Figure 1: Color heatmap
        fig = plt.figure(figsize=(6.5, 4.5))
        create_publication_heatmap(fig, df)
        pdf.savefig(fig, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
        print("  ✓ Figure 1: Color heatmap (viridis - colorblind-friendly)")
        
        # Figure 2: Grayscale heatmap
        fig = plt.figure(figsize=(6.5, 4.5))
        create_grayscale_heatmap(fig, df)
        pdf.savefig(fig, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
        print("  ✓ Figure 2: Grayscale heatmap (print journals)")
        
        # Tables
        fig = plt.figure(figsize=(8.5, 11))
        create_statistics_table(fig, df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Supplementary Tables")
        
        # Metadata
        d = pdf.infodict()
        d['Title'] = 'Zero-Shot Classification: Embedding Model Comparison'
        d['Author'] = 'Research Team'
        d['Subject'] = 'Publication Figure - F1 Score Heatmap'
        d['Keywords'] = 'Zero-Shot Classification, Sentence Embeddings, F1 Score'
        d['CreationDate'] = datetime.now()
    
    print(f"\n✅ Publication-ready PDF: {output_file}")
    print(f"\n📋 Academic Terminology:")
    print(f"   ✓ 'Embedding Model' (not 'Model' or 'Bi-encoder')")
    print(f"   ✓ Common in: ACL, EMNLP, NeurIPS, ICLR papers")
    print(f"\n📝 Suggested Caption:")
    print(f"   Figure X. Performance comparison of sentence embedding models on")
    print(f"   zero-shot text classification across seven benchmark datasets.")
    print(f"   Heat map shows Macro F1 scores (%). Darker colors indicate")
    print(f"   better performance.")
    print(f"\n📊 Results:")
    print(f"   Models: {len(df)}")
    print(f"   Datasets: {len(df.columns)}")
    print(f"   Best Model: {df.mean(axis=1).idxmax()} ({df.mean(axis=1).max():.1f}%)")
    print(f"   Easiest Dataset: {df.mean(axis=0).idxmax()} ({df.mean(axis=0).max():.1f}%)")
    print(f"   Hardest Dataset: {df.mean(axis=0).idxmin()} ({df.mean(axis=0).min():.1f}%)")

if __name__ == '__main__':
    generate_pdf()