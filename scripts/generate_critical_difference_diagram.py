"""
Generate Critical Difference Diagram (Demšar 2006)
Statistical comparison of models across datasets using Friedman test and Nemenyi post-hoc
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_results():
    """Load all experiment results and create dataset × model matrix"""
    results_dir = Path('results/raw')
    
    # Collect all results
    data = []
    for json_file in results_dir.glob('*_metrics.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                # Extract dataset name
                dataset = result.get('dataset') or result.get('dataset_name', '')
                exp_name = result.get('experiment_name', json_file.stem)
                
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
                
                # Extract model name
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
                else:
                    continue
                
                data.append({
                    'dataset': dataset,
                    'model': model,
                    'macro_f1': result['macro_f1'] * 100
                })
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
    
    df = pd.DataFrame(data)
    
    # Remove duplicates (keep best score)
    df = df.sort_values('macro_f1', ascending=False).drop_duplicates(
        subset=['dataset', 'model'], keep='first'
    )
    
    return df

def create_ranking_matrix(df):
    """Create dataset × model ranking matrix"""
    
    # Pivot to get dataset × model matrix
    pivot = df.pivot(index='dataset', columns='model', values='macro_f1')
    
    print("\n" + "="*80)
    print("F1 SCORES MATRIX (Dataset × Model)")
    print("="*80)
    print(pivot.round(1))
    
    # Rank models for each dataset (1 = best)
    ranks = pivot.rank(axis=1, ascending=False)
    
    print("\n" + "="*80)
    print("RANKING MATRIX (Dataset × Model)")
    print("="*80)
    print("1 = best, 7 = worst")
    print(ranks)
    
    # Calculate average rank for each model
    avg_ranks = ranks.mean(axis=0).sort_values()
    
    print("\n" + "="*80)
    print("AVERAGE RANKS (Lower is better)")
    print("="*80)
    for model, rank in avg_ranks.items():
        print(f"  {model:15s}: {rank:.2f}")
    
    return pivot, ranks, avg_ranks

def friedman_test(ranks):
    """Perform Friedman test"""
    from scipy.stats import friedmanchisquare
    
    # Get ranks for each model across datasets
    model_ranks = [ranks[col].values for col in ranks.columns]
    
    statistic, p_value = friedmanchisquare(*model_ranks)
    
    print("\n" + "="*80)
    print("FRIEDMAN TEST")
    print("="*80)
    print(f"Chi-square statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("✓ Significant differences detected (p < 0.05)")
        print("  → Models perform significantly differently across datasets")
    else:
        print("✗ No significant differences (p >= 0.05)")
    
    return statistic, p_value

def nemenyi_critical_distance(n_datasets, n_models, alpha=0.05):
    """Calculate critical distance for Nemenyi test"""
    from scipy.stats import studentized_range
    
    # Studentized range statistic (q_alpha)
    # For alpha=0.05, k=7 models, this is approximately 3.344
    q_alpha = studentized_range.ppf(1 - alpha, n_models, np.inf)
    
    # Critical distance formula
    cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
    
    return cd

def plot_critical_difference_diagram(avg_ranks, cd, output_file):
    """Plot Critical Difference Diagram (Demšar 2006 style) - Proper version"""
    
    # Sort models by average rank (best to worst)
    sorted_models = avg_ranks.sort_values()
    models = sorted_models.index.tolist()
    ranks = sorted_models.values
    
    print("\n" + "="*80)
    print("MODELS SORTED BY AVERAGE RANK (Best → Worst)")
    print("="*80)
    for i, (model, rank) in enumerate(zip(models, ranks), 1):
        print(f"  {i}. {model:15s}: {rank:.2f}")
    
    n_models = len(models)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Draw main rank axis
    min_rank = 1
    max_rank = n_models
    ax.plot([min_rank, max_rank], [0, 0], 'k-', linewidth=2)
    
    # Draw rank ticks and labels
    for i in range(1, n_models + 1):
        ax.plot([i, i], [-0.03, 0.03], 'k-', linewidth=2)
        ax.text(i, -0.12, str(i), ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Axis label
    ax.text((min_rank + max_rank) / 2, -0.25, 'Average Rank', 
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Draw CD bar on the axis (as a reference scale)
    cd_center = (min_rank + max_rank) / 2
    cd_x_start = cd_center - cd / 2
    cd_x_end = cd_center + cd / 2
    cd_y = 0.12
    
    # CD bar
    ax.plot([cd_x_start, cd_x_end], [cd_y, cd_y], 'k-', linewidth=2.5)
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.03, cd_y + 0.03], 'k-', linewidth=2.5)
    ax.plot([cd_x_end, cd_x_end], [cd_y - 0.03, cd_y + 0.03], 'k-', linewidth=2.5)
    ax.text(cd_center, cd_y + 0.08, f'CD = {cd:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Find cliques (groups of models not significantly different)
    # Two models are in same clique if their rank difference <= CD
    cliques = []
    used = set()
    
    for i in range(n_models):
        if i in used:
            continue
        clique = [i]
        for j in range(i + 1, n_models):
            if ranks[j] - ranks[i] <= cd:
                clique.append(j)
        if len(clique) > 1:
            cliques.append(clique)
            used.update(clique)
    
    print(f"\n📊 Found {len(cliques)} clique(s) (groups not significantly different):")
    for idx, clique in enumerate(cliques, 1):
        clique_models = [models[i] for i in clique]
        print(f"   Clique {idx}: {', '.join(clique_models)}")
    
    # Assign vertical positions to avoid label overlap
    # Use alternating top/bottom placement for closely spaced models
    y_levels = {}
    
    # Group models by horizontal proximity
    MIN_SPACING = 0.5  # Minimum rank difference to be on same level
    
    # Sort by rank and assign levels
    sorted_indices = np.argsort(ranks)
    
    for i, idx in enumerate(sorted_indices):
        if idx in y_levels:
            continue
        
        # Check if this model is in a clique
        in_clique = False
        clique_level = None
        for clique in cliques:
            if idx in clique:
                in_clique = True
                # All clique members should be on same level
                if any(j in y_levels for j in clique):
                    clique_level = y_levels[clique[0]]
                break
        
        if in_clique and clique_level is not None:
            y_levels[idx] = clique_level
            continue
        
        # Find appropriate level to avoid overlap
        # Check previous models
        level_options = [0.35, 0.53, 0.71, 0.89, 1.07]  # Multiple levels
        chosen_level = level_options[0]
        
        for level in level_options:
            overlap = False
            for prev_idx in sorted_indices[:i]:
                if prev_idx not in y_levels:
                    continue
                # Check horizontal distance and vertical level
                h_dist = abs(ranks[idx] - ranks[prev_idx])
                v_dist = abs(y_levels[prev_idx] - level)
                
                # If horizontally close and on same level, there's overlap
                if h_dist < MIN_SPACING and v_dist < 0.1:
                    overlap = True
                    break
            
            if not overlap:
                chosen_level = level
                break
        
        y_levels[idx] = chosen_level
        
        # If in clique, assign same level to all clique members
        if in_clique:
            for clique in cliques:
                if idx in clique:
                    for j in clique:
                        y_levels[j] = chosen_level
                    break
    
    # Draw models
    for i, (model, rank) in enumerate(zip(models, ranks)):
        y_pos = y_levels[i]
        
        # Vertical line from axis to label
        ax.plot([rank, rank], [0, y_pos], 'k-', linewidth=1, alpha=0.5)
        
        # Marker on axis
        ax.plot(rank, 0, 'o', markersize=9, color='steelblue', 
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # Label
        ax.text(rank, y_pos + 0.03, model, ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Draw clique lines (thick red lines connecting models in same clique)
    for clique in cliques:
        clique_ranks = [ranks[i] for i in clique]
        clique_y = y_levels[clique[0]]
        
        # Thick horizontal line
        ax.plot([min(clique_ranks), max(clique_ranks)], 
                [clique_y, clique_y], 'r-', linewidth=4, alpha=0.7, solid_capstyle='round')
    
    # Set limits
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(-0.35, max(y_levels.values()) + 0.2)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    
    # Also save as PDF
    pdf_file = output_file.replace('.png', '.pdf')
    pdf_file = pdf_file.replace('results/plots', 'reports')
    Path('reports').mkdir(exist_ok=True)
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pdf_file}")
    
    print(f"\n📊 Interpretation:")
    print(f"   • CD = {cd:.3f} (critical distance)")
    print(f"   • Red lines connect models NOT significantly different")
    print(f"   • Models without red lines ARE significantly different")
    
    plt.show()

def main():
    print("="*80)
    print("CRITICAL DIFFERENCE DIAGRAM (Demšar 2006)")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    df = load_results()
    print(f"✓ Loaded {len(df)} results")
    print(f"✓ Datasets: {df['dataset'].nunique()}")
    print(f"✓ Models: {df['model'].nunique()}")
    
    # Create ranking matrix
    pivot, ranks, avg_ranks = create_ranking_matrix(df)
    
    # Friedman test
    statistic, p_value = friedman_test(ranks)
    
    # Calculate critical distance
    n_datasets = len(ranks)
    n_models = len(ranks.columns)
    cd = nemenyi_critical_distance(n_datasets, n_models, alpha=0.05)
    
    print("\n" + "="*80)
    print("NEMENYI POST-HOC TEST")
    print("="*80)
    print(f"Critical Distance (CD): {cd:.3f}")
    print(f"Models with rank difference < {cd:.3f} are NOT significantly different")
    
    # Check pairwise differences
    print("\nPairwise comparisons:")
    sorted_models = avg_ranks.sort_values()
    for i, (model1, rank1) in enumerate(sorted_models.items()):
        for model2, rank2 in list(sorted_models.items())[i+1:]:
            diff = abs(rank1 - rank2)
            sig = "✓ Significant" if diff > cd else "✗ Not significant"
            print(f"  {model1:15s} vs {model2:15s}: Δrank={diff:.2f}  {sig}")
    
    # Plot CD diagram
    output_file = 'results/plots/critical_difference_diagram.png'
    plot_critical_difference_diagram(avg_ranks, cd, output_file)
    
    # Save ranking data
    ranks_file = 'results/tables/model_rankings.csv'
    ranks.to_csv(ranks_file)
    print(f"\n✅ Saved rankings: {ranks_file}")
    
    avg_ranks_file = 'results/tables/average_ranks.csv'
    avg_ranks.to_csv(avg_ranks_file)
    print(f"✅ Saved average ranks: {avg_ranks_file}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("• Lower average rank = better overall performance")
    print("• Red lines connect models that are NOT significantly different")
    print("• Models without red lines ARE significantly different")
    print(f"• Critical Distance (CD) = {cd:.3f}")
    print("• Based on Friedman test + Nemenyi post-hoc (α=0.05)")
    
    print("\n" + "="*80)
    print("FOR PAPER")
    print("="*80)
    print("Figure caption suggestion:")
    print("Critical difference diagram comparing the seven embedding models across")
    print("seven datasets using the Friedman test with Nemenyi post-hoc analysis")
    print("(α=0.05). Models are ranked by average performance (lower is better).")
    print("Red horizontal lines connect models that are not significantly different.")
    print(f"The critical distance (CD={cd:.2f}) indicates the minimum rank difference")
    print("required for statistical significance.")

if __name__ == '__main__':
    main()
