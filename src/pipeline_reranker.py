"""Reranker-based zero-shot classification pipeline."""

import numpy as np
from typing import List, Tuple
from src.rerankers import CrossEncoderReranker


def format_as_nli_hypothesis(label_text: str) -> str:
    """Format label description as NLI hypothesis.
    
    For zero-shot classification with cross-encoders, we use NLI format:
    - Premise: the text to classify
    - Hypothesis: "This text is about {label}"
    
    Args:
        label_text: Label description
        
    Returns:
        Formatted hypothesis string
    """
    # If label_text is already a full sentence (description mode), use it as-is
    if label_text.startswith("This "):
        return label_text
    
    # Otherwise, wrap it in NLI hypothesis format
    return f"This text is about {label_text}."


def predict_reranker(
    texts: List[str],
    label_texts: List[str],
    label_ids: List[int],
    reranker: CrossEncoderReranker,
    batch_size: int = 32,
    show_progress: bool = True,
    use_nli_format: bool = True,
) -> Tuple[List[int], List[float], np.ndarray]:
    """Predict labels using cross-encoder reranker.
    
    Args:
        texts: List of input texts to classify
        label_texts: List of label descriptions
        label_ids: Corresponding label IDs for each label text
        reranker: CrossEncoderReranker instance
        batch_size: Batch size for scoring
        show_progress: Whether to show progress bar
        use_nli_format: Whether to format labels as NLI hypotheses
        
    Returns:
        Tuple of (predictions, confidences, all_scores)
        - predictions: Predicted label IDs (n_samples,)
        - confidences: Confidence scores for predictions (n_samples,)
        - all_scores: All scores for each text-label pair (n_samples, n_labels)
    """
    n_texts = len(texts)
    n_labels = len(label_texts)
    
    # Format labels as NLI hypotheses if requested
    if use_nli_format:
        formatted_labels = [format_as_nli_hypothesis(label) for label in label_texts]
        print(f"\nUsing NLI format for reranker")
        print(f"Example: '{label_texts[0]}' -> '{formatted_labels[0]}'")
    else:
        formatted_labels = label_texts
    
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
        scores = reranker.score(text, formatted_labels, batch_size=batch_size)
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)  # Shape: (n_texts, n_labels)
    
    # Get predictions (highest scoring label)
    pred_indices = np.argmax(all_scores, axis=1)
    
    # Map indices to label IDs - convert to Python list first
    predictions = [label_ids[idx] for idx in pred_indices.tolist()]
    
    # Get confidence scores (max score for each text)
    confidences = np.max(all_scores, axis=1).tolist()
    
    return predictions, confidences, all_scores
