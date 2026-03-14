"""
Generate Academic Dataset Size Report (PDF)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime

# Set professional style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Dataset information
datasets = {
    'AG News': {
        'classes': 4,
        'original_size': 7600,
        'used_size': 1000,
        'categories': 'World, Sports, Business, Sci/Tech'
    },
    'DBPedia-14': {
        'classes': 14,
        'original_size': 70000,
        'used_size': 1000,
        'categories': 'Company, Artist, Athlete, etc.'
    },
    'Yahoo Answers': {
        'classes': 10,
        'original_size': 60000,
        'used_size': 1000,
        'categories': 'Society, Science, Health, etc.'
    },
    'Banking77': {
        'classes': 77,
        'original_size': 3080,
        'used_size': 1000,
        'categories': 'Banking domain intents'
    },
    '20 Newsgroups': {
        'classes': 20,
        'original_size': 7532,
        'used_size': 2000,
        'categories': 'comp.graphics, sci.space, etc.'
    },
    'Twitter Financial': {
        'classes': 3,
        'original_size': 3000,
        'used_size': 1000,
        'categories': 'Bearish, Bullish, Neutral'
    }
}

def create_title_page(fig):
    """Create title page"""
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.75, 'Dataset Sampling Strategy Report', 
             ha='center', va='center', fontsize=20, weight='bold')
    
    fig.text(0.5, 0.68, 'Zero-Shot Text Classification Benchmarks',
             ha='center', va='center', fontsize=14, style='italic')
    
    # Metadata
    fig.text(0.5, 0.55, f'Report Date: {datetime.now().strftime("%B %d, %Y")}',
             ha='center', va='center', fontsize=11)
    
    # Abstract box
    abstract_text = """This report presents the dataset selection and sampling methodology for 
zero-shot text classification experiments across six benchmark datasets.
Our sampling strategy (dataset.shuffle(seed=42).select(range(N))) ensures
reproducible, unbiased random sampling that is consistent with academic
standards in the field."""
    
    fig.text(0.5, 0.35, 'Summary', ha='center', va='center', 
             fontsize=12, weight='bold')
    fig.text(0.1, 0.28, abstract_text, ha='left', va='top', 
             fontsize=10, wrap=True, bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))

def create_sampling_visualization(fig):
    """Create sampling ratio visualization"""
    fig.suptitle('Dataset Sizes and Sampling Ratios', 
                 fontsize=12, weight='bold', y=0.98)
    
    # Data
    names = list(datasets.keys())
    original = [datasets[n]['original_size'] for n in names]
    used = [datasets[n]['used_size'] for n in names]
    ratios = [(u/o)*100 for u, o in zip(used, original)]
    
    # Create subplots
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3,
                         left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # 1. Original vs Used (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original, width, label='Original Test Set',
                    color='#4472C4', alpha=0.7)
    bars2 = ax1.bar(x + width/2, used, width, label='Used Samples',
                    color='#ED7D31', alpha=0.7)
    
    ax1.set_ylabel('Number of Samples (log scale)', fontsize=10)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_title('Dataset Sizes', fontsize=10, weight='bold')
    
    # 2. Usage ratios
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#70AD47' if r > 25 else '#FFC000' if r > 10 else '#5B9BD5' 
              for r in ratios]
    bars = ax2.barh(names, ratios, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Usage Ratio (%)', fontsize=10)
    ax2.set_title('Sampling Ratios', fontsize=10, weight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax2.text(ratio + 1, i, f'{ratio:.1f}%', 
                va='center', fontsize=8)
    
    # 3. Sample size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sizes, counts = np.unique(used, return_counts=True)
    ax3.bar(range(len(sizes)), counts, color='#A5A5A5', alpha=0.7)
    ax3.set_xticks(range(len(sizes)))
    ax3.set_xticklabels([f'{int(s):,}' for s in sizes])
    ax3.set_xlabel('Sample Size', fontsize=10)
    ax3.set_ylabel('Number of Datasets', fontsize=10)
    ax3.set_title('Sample Size Distribution', fontsize=10, weight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Classes vs samples
    ax4 = fig.add_subplot(gs[1, 1])
    classes = [datasets[n]['classes'] for n in names]
    scatter = ax4.scatter(classes, used, s=150, c=ratios, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidths=2)
    
    for i, name in enumerate(names):
        ax4.annotate(name, (classes[i], used[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8)
    
    ax4.set_xlabel('Number of Classes', fontsize=10)
    ax4.set_ylabel('Used Samples', fontsize=10)
    ax4.set_title('Classes vs Sample Size', fontsize=10, weight='bold')
    ax4.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Usage Ratio (%)', fontsize=8)

def create_methodology_page(fig):
    """Create methodology explanation page"""
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    fig.text(0.5, 0.95, 'Sampling Methodology', ha='center', 
             va='top', fontsize=14, weight='bold')
    
    # Sampling code
    code_text = """Sampling Implementation:

dataset = dataset.shuffle(seed=42).select(range(max_samples))

• shuffle(seed=42): Ensures reproducible random ordering
• select(range(N)): Selects first N samples after shuffling
• Result: Simple random sampling, unbiased and reproducible"""
    
    fig.text(0.1, 0.85, code_text, ha='left', va='top', fontsize=9,
             family='monospace', bbox=dict(boxstyle='round', 
             facecolor='lightgray', alpha=0.3))
    
    # Statistical justification
    stats_text = """Statistical Confidence:

For n=1,000 samples (95% confidence, p=0.5):
    Margin of Error = ±3.1%

For n=2,000 samples:
    Margin of Error = ±2.2%

These margins are acceptable for model comparison studies."""
    
    fig.text(0.1, 0.65, stats_text, ha='left', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    # Dataset table
    table_data = []
    headers = ['Dataset', 'Test Size', 'Used', 'Ratio', 'CI']
    for name, info in datasets.items():
        ratio = (info['used_size'] / info['original_size']) * 100
        ci = 1.96 * np.sqrt(0.25 / info['used_size']) * 100
        table_data.append([
            name,
            f"{info['original_size']:,}",
            f"{info['used_size']:,}",
            f"{ratio:.1f}%",
            f"±{ci:.1f}%"
        ])
    
    # Create table
    ax_table = fig.add_axes([0.1, 0.15, 0.8, 0.35])
    ax_table.axis('off')
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')

def generate_pdf():
    """Generate complete PDF report"""
    output_file = 'DATASET_SAMPLING_REPORT.pdf'
    
    print("Generating PDF report...")
    
    with PdfPages(output_file) as pdf:
        # Page 1: Title
        fig = plt.figure(figsize=(8.5, 11))
        create_title_page(fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 1: Title page")
        
        # Page 2: Visualizations
        fig = plt.figure(figsize=(8.5, 11))
        create_sampling_visualization(fig)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        print("  ✓ Page 2: Visualizations")
        
        # Page 3: Methodology
        fig = plt.figure(figsize=(8.5, 11))
        create_methodology_page(fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 3: Methodology & Table")
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Dataset Sampling Strategy Report'
        d['Author'] = 'Zero-Shot Classification Research'
        d['Subject'] = 'Dataset Size and Sampling Methodology'
        d['CreationDate'] = datetime.now()
    
    print(f"\n✅ PDF report generated: {output_file}")
    print(f"   Pages: 3")
    print(f"   Format: Letter (8.5 x 11 inches)")
    print(f"   Resolution: 300 DPI")
    print(f"   Content: Academic standard with visualizations and methodology")

if __name__ == '__main__':
    generate_pdf()