"""Classification pipelines for zero-shot classification."""

import numpy as np
from typing import List, Tuple
from src.encoders import BiEncoder, compute_similarities


def predict_biencoder(
    texts: List[str],
    label_texts: List[str],
    label_ids: List[int],
    encoder: BiEncoder,
    normalize: bool = True,
    batch_size: int = 32,
) -> Tuple[List[int], List[float], np.ndarray]:
    """Predict using bi-encoder only.
    
    Args:
        texts: Input texts to classify
        label_texts: List of label text representations
        label_ids: Corresponding label IDs for each label text
        encoder: Bi-encoder model
        normalize: Whether to normalize embeddings
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (predictions, confidences, similarity_matrix)
    """
    print(f"Encoding {len(texts)} texts...")
    text_emb = encoder.encode(texts, batch_size=batch_size, normalize=normalize, show_progress=True)
    
    print(f"Encoding {len(label_texts)} label texts...")
    label_emb = encoder.encode(label_texts, batch_size=batch_size, normalize=normalize, show_progress=False)
    
    print("Computing similarities...")
    sim = compute_similarities(text_emb, label_emb)
    
    # Get predictions
    pred_indices = sim.argmax(axis=1)
    predictions = [label_ids[i] for i in pred_indices]
    
    # Get confidence scores
    confidences = sim.max(axis=1).tolist()
    
    return predictions, confidences, sim