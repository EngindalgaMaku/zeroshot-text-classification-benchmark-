"""
Generate Publication-Quality Heatmap
High-resolution PNG + PDF for journal submission
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import numpy as np

# Publication-quality settings (IEEE/Nature/Elsevier standard)
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
    'savefig.dpi': 600,  # Very high DPI for journal
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

def load_results():
    """Load all results"""
    results_dir = Path('results/raw')
    data = []
    
    for json_file in results_dir.glob('*_metrics.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                dataset = result.get('dataset') or result.get('dataset_name', '')
                exp_name = result.get('experiment_name', json_file.stem)
                
                # Clean dataset names
                dataset_map = {
                    'ag_news': 'AG News',
                    'dbpedia_14': 'DBpedia',
                    'yahoo_answers_topics': 'Yahoo',
                    'banking77': 'Banking77',
                    'zeroshot/twitter-financial-news-sentiment': 'Twitter',
                    'SetFit/20_newsgroups': '20 Newsgroups',
                    'go_emotions': 'GoEmotions'
                }
                
                for key, val in dataset_map.items():
                    if key in dataset or key in exp_name:
                        dataset = val
                        break
                
                # Extract model
                model = None
                if "instructor" in exp_name.lower():
                    model = "INSTRUCTOR"
                elif "qwen" in exp_name.lower():
                    model = "Qwen3"
                elif "snowflake" in exp_name.lower() or "arctic" in exp_name.lower():
                    model = "Snowflake"
                elif "jina" in exp_name.lower():
                    model = "Jina-v5"
                elif "bge" in exp_name.lower():
                    model = "BGE-M3"
                elif "e5" in exp_name.lower():
                    model = "E5-large"
                elif "mpnet" in exp_name.lower():
                    model = "MPNet"
                
                if model and dataset:
                    data.append({
                        'model': model,
                        'dataset': dataset,
                        'macro_f1': result['macro_f1'] * 100
                    })
        except Exception as e:
            print(f"⚠️  Error: {json_file.name}: {e}")
    
    df = pd.DataFrame(data)
    
    # Remove duplicates
    df = df.sort_values('macro_f1', ascending=False).drop_duplicates(
        subset=['model', 'dataset'], keep='first'
    )
    
    return df

def create_heatmap():
    """Create publication-quality heatmap"""
    
    print("Loading results...")
    df = load_results()
    
    # Pivot
    pivot = df.pivot(index='model', columns='dataset', values='macro_f1')
    
    # 1️⃣ Reorder columns by difficulty (easy → hard)
    col_order = ['AG News', 'DBpedia', 'Banking77', '20 Newsgroups', 'Twitter', 'Yahoo', 'GoEmotions']
    pivot = pivot[[col for col in col_order if col in pivot.columns]]
    
    # 2️⃣ Reorder rows by average performance (best → worst)
    pivot['_avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('_avg', ascending=False)
    pivot = pivot.drop('_avg', axis=1)
    
    print(f"\n✓ Matrix: {pivot.shape[0]} models × {pivot.shape[1]} datasets")
    data_min = pivot.min().min()
    data_max = pivot.max().max()
    print(f"✓ Value range: {data_min:.1f}% - {data_max:.1f}%")
    
    # Find best score per dataset (for highlighting)
    best_per_dataset = pivot.idxmax(axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Colormap: viridis (colorblind-friendly, standard in science)
    cmap = 'viridis'
    
    # 4️⃣ Adjust color scale to actual data range for better contrast
    # Use 0 as minimum (standard for percentage scales)
    # Use actual max rounded up to nearest 5
    vmin = 0
    vmax = int(np.ceil(data_max / 5) * 5)  # Round up to nearest 5
    
    print(f"✓ Color scale: {vmin} - {vmax}%")
    print(f"✓ This maximizes contrast for range {data_min:.1f}% - {data_max:.1f}%")
    
    # Create heatmap
    hm = sns.heatmap(pivot, 
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={
                    'shrink': 0.85,
                    'aspect': 25,
                    'pad': 0.02
                },
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                square=False,
                annot_kws={'size': 10, 'weight': 'normal'}
    )
    
    # No highlighting - clean heatmap
    # Color gradient is sufficient to show best scores
    
    # Adjust colorbar label - MUST be done after heatmap creation
    cbar = hm.collections[0].colorbar
    cbar.set_label('Macro-F1 (%)', fontsize=9, rotation=270, labelpad=12)
    
    # Adjust colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Macro-F1 (%)', fontsize=10, rotation=270, labelpad=15)
    
    # Labels (no title - goes in caption)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # 5️⃣ Rotate x labels at 35° for better readability
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=35, 
                       ha='right',
                       rotation_mode='anchor')
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Save high-resolution PNG
    png_file = 'results/plots/heatmap_publication.png'
    plt.savefig(png_file, dpi=600, bbox_inches='tight', format='png')
    print(f"\n✅ Saved PNG (600 DPI): {png_file}")
    
    # Save PDF (vector format)
    pdf_file = 'reports/heatmap_publication.pdf'
    Path('reports').mkdir(exist_ok=True)
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved PDF (vector): {pdf_file}")
    
    # Also save as EPS (some journals prefer this)
    eps_file = 'reports/heatmap_publication.eps'
    plt.savefig(eps_file, dpi=300, bbox_inches='tight', format='eps')
    print(f"✅ Saved EPS (vector): {eps_file}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("HEATMAP STATISTICS")
    print("="*80)
    print(f"Models: {len(pivot)}")
    print(f"Datasets: {len(pivot.columns)}")
    print(f"Min F1: {pivot.min().min():.1f}%")
    print(f"Max F1: {pivot.max().max():.1f}%")
    print(f"Mean F1: {pivot.mean().mean():.1f}%")
    
    print("\n" + "="*80)
    print("FOR PAPER")
    print("="*80)
    print("Figure caption suggestion:")
    print()
    print("Figure X. Performance heatmap showing Macro-F1 scores (%) for seven")
    print("embedding models across seven benchmark datasets. Darker colors indicate")
    print("higher performance. Models are ordered by average performance (top to bottom).")
    print()
    print("File formats provided:")
    print("  • PNG (600 DPI) - for online/color viewing")
    print("  • PDF (vector) - for print publication")
    print("  • EPS (vector) - alternative for some journals")

if __name__ == '__main__':
    print("="*80)
    print("PUBLICATION-QUALITY HEATMAP GENERATOR")
    print("="*80)
    create_heatmap()
