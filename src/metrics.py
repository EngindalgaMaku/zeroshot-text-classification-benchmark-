"""Evaluation metrics for classification."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
from typing import List, Dict, Any


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_confidence_metrics(
    y_true: List[int],
    y_pred: List[int],
    confidences: List[float],
) -> Dict[str, Any]:
    """Compute confidence-based metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Confidence scores
        
    Returns:
        Dictionary of confidence metrics
    """
    correct = np.array([yt == yp for yt, yp in zip(y_true, y_pred)])
    confs = np.array(confidences)
    
    metrics = {
        "mean_confidence": float(confs.mean()),
        "std_confidence": float(confs.std()),
        "mean_confidence_correct": float(confs[correct].mean()) if correct.any() else 0.0,
        "mean_confidence_incorrect": float(confs[~correct].mean()) if (~correct).any() else 0.0,
    }
    
    return metrics


def analyze_errors(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    confidences: List[float],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Analyze prediction errors.
    
    Args:
        texts: Input texts
        y_true: True labels
        y_pred: Predicted labels
        confidences: Confidence scores
        top_k: Number of top errors to return
        
    Returns:
        List of error analysis dictionaries
    """
    errors = []
    
    for text, yt, yp, conf in zip(texts, y_true, y_pred, confidences):
        if yt != yp:
            errors.append({
                "text": text[:200],  # Truncate for readability
                "true_label": yt,
                "pred_label": yp,
                "confidence": conf,
            })
    
    # Sort by confidence (high confidence errors are most interesting)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    
    return errors[:top_k]