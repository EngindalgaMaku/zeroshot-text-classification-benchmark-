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
        # For NLI models, we need to score each (text, hypothesis) pair separately
        # and extract the entailment probability for each label
        label_scores = []
        
        for label_text in formatted_labels:
            # Create (premise, hypothesis) pair
            pair = (text, label_text)
            
            # Get scores: [contradiction, entailment, neutral] or similar
            scores = reranker.model.predict([pair])[0]  # Shape: (3,)
            
            # Apply softmax to get probabilities
            from scipy.special import softmax
            probs = softmax(scores)
            
            # Extract entailment probability (index 1 for most NLI models)
            # Check model config to be sure, but typically: [contradiction, entailment, neutral]
            entailment_prob = probs[1]
            label_scores.append(entailment_prob)
        
        all_scores.append(label_scores)
    
    all_scores = np.array(all_scores)  # Shape: (n_texts, n_labels)
    
    print(f"Final all_scores shape: {all_scores.shape}")
    print(f"Sample entailment probs (first text, first 5 labels): {all_scores[0, :5]}")
    
    # Get predictions (highest entailment probability)
    pred_indices = np.argmax(all_scores, axis=1)  # Shape: (n_texts,)
    print(f"First 10 predictions (indices): {pred_indices[:10]}")
    
    # Map indices to label IDs using numpy indexing
    label_ids_array = np.array(label_ids)
    predictions = label_ids_array[pred_indices].tolist()
    
    # Get confidence scores (max entailment probability for each text)
    confidences = np.max(all_scores, axis=1).tolist()
    print(f"Sample confidences (first 5): {confidences[:5]}")
    
    return predictions, confidences, all_scores
