"""Utility functions."""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


def ensure_dir(path: str):
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    path_obj = Path(path)
    
    # If path is a symlink, resolve it first
    if path_obj.is_symlink():
        path_obj = path_obj.resolve()
    
    # Create directory if it doesn't exist
    path_obj.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: Dict[str, Any], output_dir: str, experiment_name: str):
    """Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
        experiment_name: Experiment name
    """
    ensure_dir(output_dir)
    path = Path(output_dir) / f"{experiment_name}_metrics.json"
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"Metrics saved to: {path}")


def save_predictions(
    rows: List[Dict[str, Any]],
    output_dir: str,
    experiment_name: str,
):
    """Save predictions to CSV file.
    
    Args:
        rows: List of prediction dictionaries
        output_dir: Output directory
        experiment_name: Experiment name
    """
    ensure_dir(output_dir)
    path = Path(output_dir) / f"{experiment_name}_predictions.csv"
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    
    print(f"Predictions saved to: {path}")


def timestamp() -> str:
    """Get current timestamp string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print formatted metrics summary.
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:   {metrics['macro_recall']:.4f}")
    
    if "mean_confidence" in metrics:
        print(f"\nMean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"Confidence (Correct): {metrics['mean_confidence_correct']:.4f}")
        print(f"Confidence (Incorrect): {metrics['mean_confidence_incorrect']:.4f}")
    
    print("="*50 + "\n")