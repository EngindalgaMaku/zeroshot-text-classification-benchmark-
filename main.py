"""Main entry point for experiments."""

import argparse
from src.config import load_config
from src.runner import run_experiment


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
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("Zero-Shot Text Classification Experiment")
    print("="*70)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Run experiment
    metrics = run_experiment(cfg)
    
    print("\nAll done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()