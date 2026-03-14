"""Bi-encoder models for text embedding."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class BiEncoder:
    """Wrapper for bi-encoder models using sentence-transformers."""
    
    def __init__(self, model_name: str, device: str = None):
        """Initialize bi-encoder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda/cpu), auto-detected if None
        """
        print(f"Loading bi-encoder: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        print(f"Model loaded on device: {self.model.device}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        a: First set of embeddings (n x d)
        b: Second set of embeddings (m x d)
        
    Returns:
        Similarity matrix (n x m)
    """
    return np.matmul(a, b.T)


def compute_similarities(
    text_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute similarity scores between texts and labels.
    
    Args:
        text_embeddings: Text embeddings (n x d)
        label_embeddings: Label embeddings (m x d)
        
    Returns:
        Similarity matrix (n x m)
    """
    return cosine_similarity_matrix(text_embeddings, label_embeddings)