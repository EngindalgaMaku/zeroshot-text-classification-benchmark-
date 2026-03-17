"""Compare experiment results for any dataset."""
import json
import argparse
from pathlib import Path

def compare_results(dataset_name):
    """Compare results for a given dataset."""
    results_dir = Path("results/raw")
    
    experiments = {
        "Manual (description)": f"exp_{dataset_name}_mpnet_metrics.json",
        "Generated L2": f"exp_{dataset_name}_mpnet_l2_metrics.json",
        "Generated L3": f"exp_{dataset_name}_mpnet_l3_metrics.json",
    }
    
    print(f"=== {dataset_name.upper()} Results Comparison ===\n")
    print(f"{'Method':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 65)
    
    for name, filename in experiments.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            acc = data.get('accuracy', 0)
            macro_f1 = data.get('macro_f1', 0)
            weighted_f1 = data.get('weighted_f1', 0)
            print(f"{name:<25} {acc:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f}")
        else:
            print(f"{name:<25} {'NOT RUN':<12} {'NOT RUN':<12} {'NOT RUN':<12}")
    
    print("\n" + "=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experiment results for a dataset")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., banking77, ag_news)")
    args = parser.parse_args()
    
    compare_results(args.dataset)
