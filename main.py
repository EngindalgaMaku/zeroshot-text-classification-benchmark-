"""Main entry point for experiments."""

import argparse
import random
import numpy as np
import torch
from src.config import load_config
from src.runner import run_experiment


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Text Classification with Modern Embedding and Reranking Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments if results already exist (useful when adding new models)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("\n" + "="*70)
    print("Zero-Shot Text Classification Experiment")
    print("="*70)
    print("🎲 Random seed: 42 (for reproducibility)")
    
    if args.skip_existing:
        print("📋 Mode: Skip existing results")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Run experiment
    metrics = run_experiment(cfg, skip_existing=args.skip_existing)
    
    print("\nAll done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()