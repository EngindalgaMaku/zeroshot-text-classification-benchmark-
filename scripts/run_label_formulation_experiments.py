"""
Run label formulation experiments comparing name_only vs description label modes.

This script executes experiments for AG News, Banking77, and GoEmotions datasets
with multiple models to analyze the impact of label formulation on performance.
"""

import subprocess
import sys
from pathlib import Path
import time

# Define all label formulation experiment configs
LABEL_FORMULATION_CONFIGS = [
    # AG News experiments (name_only variants)
    "experiments/label_formulation/exp_agnews_nomic_name_only.yaml",
    "experiments/label_formulation/exp_agnews_instructor_name_only.yaml",
    "experiments/label_formulation/exp_agnews_mpnet_name_only.yaml",
    "experiments/label_formulation/exp_agnews_jina_v3_name_only.yaml",
    "experiments/label_formulation/exp_agnews_qwen3_name_only.yaml",
    
    # Banking77 experiments (name_only variants)
    "experiments/label_formulation/exp_banking77_nomic_name_only.yaml",
    "experiments/label_formulation/exp_banking77_instructor_name_only.yaml",
    "experiments/label_formulation/exp_banking77_jina_name_only.yaml",
    
    # GoEmotions experiments (name_only variants)
    "experiments/label_formulation/exp_goemotions_nomic_name_only.yaml",
    "experiments/label_formulation/exp_goemotions_instructor_name_only.yaml",
    "experiments/label_formulation/exp_goemotions_qwen3_name_only.yaml",
]


def run_experiment(config_path: str) -> bool:
    """
    Run a single experiment.
    
    Args:
        config_path: Path to experiment config file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running: {config_path}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--config", config_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ Successfully completed: {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {config_path}")
        print(f"Error: {e}")
        return False


def main():
    """Run all label formulation experiments."""
    print("\n" + "="*70)
    print("LABEL FORMULATION ANALYSIS - BATCH EXPERIMENT RUNNER")
    print("="*70)
    print(f"\nTotal experiments to run: {len(LABEL_FORMULATION_CONFIGS)}")
    print("\nDatasets: AG News, Banking77, GoEmotions")
    print("Label modes: name_only (comparing against existing description results)")
    print("\n" + "="*70)
    
    # Verify all config files exist
    missing_configs = []
    for config in LABEL_FORMULATION_CONFIGS:
        if not Path(config).exists():
            missing_configs.append(config)
    
    if missing_configs:
        print("\n⚠ WARNING: The following config files are missing:")
        for config in missing_configs:
            print(f"  - {config}")
        print("\nPlease create these configs before running experiments.")
        return
    
    # Run experiments
    start_time = time.time()
    successful = []
    failed = []
    
    for i, config in enumerate(LABEL_FORMULATION_CONFIGS, 1):
        print(f"\n[{i}/{len(LABEL_FORMULATION_CONFIGS)}] Processing: {config}")
        
        if run_experiment(config):
            successful.append(config)
        else:
            failed.append(config)
            # Continue with remaining experiments even if one fails
            print(f"⚠ Continuing with remaining experiments...")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(LABEL_FORMULATION_CONFIGS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    
    if failed:
        print("\n⚠ Failed experiments:")
        for config in failed:
            print(f"  - {config}")
    
    print("\n" + "="*70)
    print("Label formulation experiments complete!")
    print("Next step: Run analysis script to compare name_only vs description results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
