"""Cross-encoder reranker models."""

from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Union


class CrossEncoderReranker:
    """Wrapper for cross-encoder reranker models."""
    
    def __init__(self, model_name: str, device: str = None):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda/cpu), auto-detected if None
        """
        print(f"Loading reranker: {model_name}")
        self.model_name = model_name
        # Some models require trust_remote_code=True (e.g., Jina reranker)
        self.model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        print(f"Reranker loaded")
    
    def score(
        self,
        query: Union[str, List[str]],
        candidates: Union[List[str], List[List[str]]],
        batch_size: int = 32,
    ) -> Union[List[float], List[List[float]]]:
        """Score query-candidate pairs.
        
        Args:
            query: Single query or list of queries
            candidates: List of candidates (single query) or list of lists (multiple queries)
            batch_size: Batch size for scoring
            
        Returns:
            Scores for each query-candidate pair
        """
        if isinstance(query, str):
            # Single query, multiple candidates
            pairs = [[query, c] for c in candidates]
            scores = self.model.predict(pairs, batch_size=batch_size)
            return scores.tolist()
        else:
            # Multiple queries, multiple candidates per query
            all_scores = []
            for q, cands in zip(query, candidates):
                pairs = [[q, c] for c in cands]
                scores = self.model.predict(pairs, batch_size=batch_size)
                all_scores.append(scores.tolist())
            return all_scores
    
    def rank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = None,
    ) -> List[tuple]:
        """Rank candidates by relevance to query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Return only top-k results (None for all)
            
        Returns:
            List of (candidate_index, score) tuples, sorted by score
        """
        scores = self.score(query, candidates)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked