"""Generate Main Benchmark Results Table (Table 1)"""
import pandas as pd
import json
from pathlib import Path

def load_results():
    """Load all results and create model × dataset table"""
    results_dir = Path('results/raw')
    data = []
    
    for json_file in results_dir.glob('*_metrics.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                # Extract dataset
                dataset = result.get('dataset') or result.get('dataset_name', '')
                exp_name = result.get('experiment_name', json_file.stem)
                
                # Clean dataset names
                dataset_map = {
                    'ag_news': 'AG News',
                    'dbpedia_14': 'DBpedia',
                    'yahoo_answers_topics': 'Yahoo',
                    'banking77': 'Banking77',
                    'zeroshot/twitter-financial-news-sentiment': 'Twitter',
                    'SetFit/20_newsgroups': '20NG',
                    'go_emotions': 'GoEmotions'
                }
                
                for key, val in dataset_map.items():
                    if key in dataset or key in exp_name:
                        dataset = val
                        break
                
                # Extract model
                model = None
                if "instructor" in exp_name.lower():
                    model = "INSTRUCTOR-large"
                elif "qwen" in exp_name.lower():
                    model = "Qwen3-Embedding-8B"
                elif "snowflake" in exp_name.lower() or "arctic" in exp_name.lower():
                    model = "snowflake-arctic-embed-m"
                elif "jina" in exp_name.lower():
                    model = "jina-embeddings-v5-text-nano"
                elif "bge" in exp_name.lower():
                    model = "bge-m3"
                elif "e5" in exp_name.lower():
                    model = "multilingual-e5-large"
                elif "mpnet" in exp_name.lower():
                    model = "all-mpnet-base-v2"
                
                if model and dataset:
                    data.append({
                        'model': model,
                        'dataset': dataset,
                        'macro_f1': result['macro_f1'] * 100
                    })
        except Exception as e:
            print(f"⚠️  Error: {json_file.name}: {e}")
    
    df = pd.DataFrame(data)
    
    # Remove duplicates (keep best)
    df = df.sort_values('macro_f1', ascending=False).drop_duplicates(
        subset=['model', 'dataset'], keep='first'
    )
    
    return df

def create_table():
    """Create main results table"""
    df = load_results()
    
    # Pivot to model × dataset
    pivot = df.pivot(index='model', columns='dataset', values='macro_f1')
    
    # Reorder columns
    col_order = ['20NG', 'AG News', 'Banking77', 'DBpedia', 'GoEmotions', 'Twitter', 'Yahoo']
    pivot = pivot[[col for col in col_order if col in pivot.columns]]
    
    # Reorder rows
    row_order = [
        'INSTRUCTOR-large',
        'Qwen3-Embedding-8B',
        'all-mpnet-base-v2',
        'jina-embeddings-v5-text-nano',
        'multilingual-e5-large',
        'bge-m3',
        'snowflake-arctic-embed-m'
    ]
    pivot = pivot.reindex([row for row in row_order if row in pivot.index])
    
    # Round to 1 decimal
    pivot = pivot.round(1)
    
    # Add average column
    pivot['Average'] = pivot.mean(axis=1).round(1)
    
    print("\n" + "="*100)
    print("TABLE 1 — MAIN BENCHMARK RESULTS")
    print("="*100)
    print("\nMetric: Macro-F1 (%)\n")
    print(pivot.to_string())
    
    # Save as CSV
    pivot.to_csv('results/tables/main_results_table.csv')
    print(f"\n✅ Saved: results/tables/main_results_table.csv")
    
    # Save as LaTeX
    latex = pivot.to_latex(float_format="%.1f")
    with open('results/tables/main_results_table.tex', 'w') as f:
        f.write(latex)
    print(f"✅ Saved: results/tables/main_results_table.tex")
    
    # Print statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    print(f"\nBest model (average): {pivot['Average'].idxmax()} ({pivot['Average'].max():.1f}%)")
    print(f"Worst model (average): {pivot['Average'].idxmin()} ({pivot['Average'].min():.1f}%)")
    
    # Best per dataset
    print("\nBest model per dataset:")
    for col in pivot.columns[:-1]:  # Exclude Average
        best_model = pivot[col].idxmax()
        best_score = pivot[col].max()
        print(f"  {col:15s}: {best_model:30s} ({best_score:.1f}%)")
    
    return pivot

if __name__ == '__main__':
    create_table()
