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
    
    # Convert to numpy array
    all_scores = np.array(all_scores)
    
    print(f"Raw scores shape: {all_scores.shape}")
    
    # NLI models return 3 scores: [contradiction, entailment, neutral]
    # For zero-shot classification, we use: entailment - contradiction
    # This gives us a relative score that discriminates between labels
    if all_scores.ndim == 3 and all_scores.shape[2] == 3:
        print("NLI model detected - using entailment-contradiction difference")
        
        # Extract contradiction (index 0) and entailment (index 1)
        contradiction_scores = all_scores[:, :, 0]
        entailment_scores = all_scores[:, :, 1]
        
        # Use the difference: entailment - contradiction
        # Higher difference = more likely to be this label
        all_scores = entailment_scores - contradiction_scores
        
        print(f"Sample scores (first text, first 5 labels): {all_scores[0, :5]}")
    
    # Ensure 2D: (n_texts, n_labels)
    if all_scores.ndim != 2:
        raise ValueError(f"Expected 2D scores array after processing, got shape {all_scores.shape}")
    
    print(f"Final all_scores shape: {all_scores.shape}")
    
    # Get predictions (highest scoring label)
    pred_indices = np.argmax(all_scores, axis=1)  # Shape: (n_texts,)
    print(f"First 10 predictions (indices): {pred_indices[:10]}")
    
    # Map indices to label IDs using numpy indexing
    label_ids_array = np.array(label_ids)
    predictions = label_ids_array[pred_indices].tolist()
    
    # Get confidence scores (max score for each text)
    confidences = np.max(all_scores, axis=1).tolist()
    
    return predictions, confidences, all_scores
