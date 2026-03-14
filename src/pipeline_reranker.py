"""Reranker-based zero-shot classification pipeline."""

import numpy as np
from typing import List, Tuple
from src.rerankers import CrossEncoderReranker


def predict_reranker(
    texts: List[str],
    label_texts: List[str],
    reranker: CrossEncoderReranker,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels using cross-encoder reranker.
    
    Args:
        texts: List of input texts to classify
        label_texts: List of label descriptions
        reranker: CrossEncoderReranker instance
        batch_size: Batch size for scoring
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (predictions, confidences, all_scores)
        - predictions: Predicted label indices (n_samples,)
        - confidences: Confidence scores for predictions (n_samples,)
        - all_scores: All scores for each text-label pair (n_samples, n_labels)
    """
    n_texts = len(texts)
    n_labels = len(label_texts)
    
    print(f"\nScoring {n_texts} texts against {n_labels} labels...")
    
    # Score all text-label pairs
    all_scores = []
    
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(texts, desc="Scoring texts")
        except ImportError:
            iterator = texts
    else:
        iterator = texts
    
    for text in iterator:
        # Score this text against all labels
        scores = reranker.score(text, label_texts, batch_size=batch_size)
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)  # Shape: (n_texts, n_labels)
    
    # Get predictions (highest scoring label)
    predictions = np.argmax(all_scores, axis=1)
    
    # Get confidence scores (max score for each text)
    confidences = np.max(all_scores, axis=1)
    
    return predictions, confidences, all_scores
