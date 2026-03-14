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
        y_true: True labels (can be multi-label lists or single integers)
        y_pred: Predicted labels (single integers)
        
    Returns:
        Dictionary of metrics
    """
    # Check if multi-label (GoEmotions case)
    is_multilabel = isinstance(y_true[0], (list, tuple, np.ndarray))
    
    if is_multilabel:
        # For multi-label: convert to single label by taking first label
        # This is a simplification for zero-shot classification
        y_true_single = [labels[0] if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) > 0 else labels 
                         for labels in y_true]
    else:
        y_true_single = y_true
    
    metrics = {
        "accuracy": float(accuracy_score(y_true_single, y_pred)),
        "macro_f1": float(f1_score(y_true_single, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true_single, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true_single, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_single, y_pred, average="macro", zero_division=0)),
    }
    
    # Per-class metrics
    report = classification_report(y_true_single, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true_single, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_confidence_metrics(
    y_true: List[int],
    y_pred: List[int],
    confidences: List[float],
) -> Dict[str, Any]:
    """Compute confidence-based metrics.
    
    Args:
        y_true: True labels (can be multi-label lists or single integers)
        y_pred: Predicted labels
        confidences: Confidence scores
        
    Returns:
        Dictionary of confidence metrics
    """
    # Check if multi-label (GoEmotions case)
    is_multilabel = isinstance(y_true[0], (list, tuple, np.ndarray))
    
    if is_multilabel:
        # For multi-label: convert to single label by taking first label
        y_true_single = [labels[0] if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) > 0 else labels 
                         for labels in y_true]
    else:
        y_true_single = y_true
    
    correct = np.array([yt == yp for yt, yp in zip(y_true_single, y_pred)])
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
        y_true: True labels (can be multi-label lists or single integers)
        y_pred: Predicted labels
        confidences: Confidence scores
        top_k: Number of top errors to return
        
    Returns:
        List of error analysis dictionaries
    """
    # Check if multi-label (GoEmotions case)
    is_multilabel = isinstance(y_true[0], (list, tuple, np.ndarray))
    
    if is_multilabel:
        # For multi-label: convert to single label by taking first label
        y_true_single = [labels[0] if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) > 0 else labels 
                         for labels in y_true]
    else:
        y_true_single = y_true
    
    errors = []
    
    for text, yt, yp, conf in zip(texts, y_true_single, y_pred, confidences):
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