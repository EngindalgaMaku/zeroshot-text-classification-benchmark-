"""Run L1/L2/L3 label experiments with progress tracking.

This script runs experiments sequentially and shows results after each completion.
"""

import subprocess
import json
from pathlib import Path
import sys


def run_experiment(config_path):
    """Run a single experiment and return metrics."""
    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print('='*80)
    
    result = subprocess.run(
        ["python", "main.py", "--config", str(config_path)],
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ FAILED: {config_path.name}")
        return None
    
    # Load and return metrics
    exp_name = config_path.stem  # Remove .yaml
    metrics_file = Path("results/llm_descriptions") / f"{exp_name}_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def show_comparison(l1_metrics, l2_metrics, l3_metrics, dataset, model):
    """Show L1/L2/L3 comparison."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {dataset} - {model}")
    print('='*80)
    
    if l1_metrics:
        print(f"L1 (name_only):     Acc={l1_metrics['accuracy']:.4f}  F1={l1_metrics['macro_f1']:.4f}")
    else:
        print("L1 (name_only):     ❌ FAILED")
    
    if l2_metrics:
        print(f"L2 (LLM single):    Acc={l2_metrics['accuracy']:.4f}  F1={l2_metrics['macro_f1']:.4f}")
    else:
        print("L2 (LLM single):    ❌ FAILED")
    
    if l3_metrics:
        print(f"L3 (LLM multi):     Acc={l3_metrics['accuracy']:.4f}  F1={l3_metrics['macro_f1']:.4f}")
    else:
        print("L3 (LLM multi):     ❌ FAILED")
    
    # Show improvements
    if l1_metrics and l2_metrics:
        l2_gain = l2_metrics['accuracy'] - l1_metrics['accuracy']
        print(f"\n📊 L2 vs L1: {l2_gain:+.4f} ({l2_gain*100:+.2f}%)")
    
    if l1_metrics and l3_metrics:
        l3_gain = l3_metrics['accuracy'] - l1_metrics['accuracy']
        print(f"📊 L3 vs L1: {l3_gain:+.4f} ({l3_gain*100:+.2f}%)")
    
    if l2_metrics and l3_metrics:
        l3_l2_gain = l3_metrics['accuracy'] - l2_metrics['accuracy']
        print(f"📊 L3 vs L2: {l3_l2_gain:+.4f} ({l3_l2_gain*100:+.2f}%)")
    
    print('='*80 + '\n')


def main():
    """Run all L1/L2/L3 experiments with progress tracking."""
    
    config_dir = Path("experiments/llm_descriptions")
    
    # Group configs by dataset and model
    configs_by_group = {}
    
    for config_path in sorted(config_dir.glob("*.yaml")):
        # Parse filename: dataset_model_level.yaml
        parts = config_path.stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        
        base_name = parts[0]  # dataset_model
        level = parts[1]      # l1, l2, or l3
        
        if base_name not in configs_by_group:
            configs_by_group[base_name] = {}
        
        configs_by_group[base_name][level] = config_path
    
    print(f"\n{'='*80}")
    print(f"LABEL SEMANTICS EXPERIMENTS")
    print(f"Total groups: {len(configs_by_group)}")
    print('='*80 + '\n')
    
    completed = 0
    total = len(configs_by_group)
    
    for base_name, configs in sorted(configs_by_group.items()):
        completed += 1
        
        # Extract dataset and model from base_name
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            dataset = parts[0]
            model = parts[1]
        else:
            dataset = base_name
            model = "unknown"
        
        print(f"\n{'#'*80}")
        print(f"# GROUP {completed}/{total}: {dataset} - {model}")
        print(f"{'#'*80}\n")
        
        # Run L1, L2, L3 in order
        l1_metrics = run_experiment(configs['l1']) if 'l1' in configs else None
        l2_metrics = run_experiment(configs['l2']) if 'l2' in configs else None
        l3_metrics = run_experiment(configs['l3']) if 'l3' in configs else None
        
        # Show comparison
        show_comparison(l1_metrics, l2_metrics, l3_metrics, dataset, model)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print('='*80 + '\n')


if __name__ == "__main__":
    main()