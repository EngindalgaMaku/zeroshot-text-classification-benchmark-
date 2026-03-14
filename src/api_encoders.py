"""API-based embedding models."""

import requests
import numpy as np
from typing import List, Union
import os


class JinaAPIEncoder:
    """Jina AI Embeddings API wrapper."""
    
    def __init__(self, model_name: str = "jina-embeddings-v3", api_key: str = None):
        """Initialize Jina API encoder.
        
        Args:
            model_name: Jina model name
            api_key: Jina API key (or set JINA_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        
        if not self.api_key:
            raise ValueError("Jina API key required. Set JINA_API_KEY or pass api_key parameter.")
        
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Initialized Jina API: {model_name}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 100,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts using Jina API.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for API calls
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress (ignored for API)
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": self.model_name,
                "input": batch,
                "encoding_type": "float"
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Jina API error: {response.status_code} - {response.text}")
            
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)
            
            if show_progress and len(texts) > batch_size:
                print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        embeddings = np.array(all_embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return embeddings


class JinaAPIReranker:
    """Jina AI Reranker API wrapper."""
    
    def __init__(self, model_name: str = "jina-reranker-v2-base-multilingual", api_key: str = None):
        """Initialize Jina API reranker.
        
        Args:
            model_name: Jina reranker model name
            api_key: Jina API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        
        if not self.api_key:
            raise ValueError("Jina API key required.")
        
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Initialized Jina Reranker API: {model_name}")
    
    def score(
        self,
        query: Union[str, List[str]],
        candidates: Union[List[str], List[List[str]]],
        batch_size: int = 32,
    ) -> Union[List[float], List[List[float]]]:
        """Score query-candidate pairs.
        
        Args:
            query: Single query or list of queries
            candidates: List of candidates or list of lists
            batch_size: Not used for API
            
        Returns:
            Scores for each query-candidate pair
        """
        if isinstance(query, str):
            # Single query
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": candidates,
                "top_n": len(candidates)
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Jina Rerank API error: {response.status_code}")
            
            result = response.json()
            # Sort by index to maintain original order
            sorted_results = sorted(result["results"], key=lambda x: x["index"])
            scores = [item["relevance_score"] for item in sorted_results]
            
            return scores
        else:
            # Multiple queries - process one by one
            all_scores = []
            for q, cands in zip(query, candidates):
                scores = self.score(q, cands)
                all_scores.append(scores)
            return all_scores