"""Bi-encoder models for text embedding.

Supports:
- Standard sentence-transformers models
- Jina embeddings with explicit task selection
- INSTRUCTOR models with instruction pairs

- Optional GTE custom backend via transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class _TransformersEncoder:
    """Custom transformers-based encoder backend for models that are not
    safely or optimally handled by SentenceTransformer directly.
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        trust_remote_code: bool = False,
        pooling: str = "cls",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.max_length = max_length

        print(f"Loading custom transformers encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        print(f"Custom model loaded on device: {self.device}")

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
        text_type: str = "text",
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Batches")
            except Exception:
                pass

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            outputs = self.model(**tokenized, return_dict=True)

            if self.pooling == "cls":
                emb = outputs.last_hidden_state[:, 0]
            elif self.pooling == "mean":
                last_hidden = outputs.last_hidden_state
                attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                masked = last_hidden * attention_mask
                emb = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


class BiEncoder:
    """Wrapper for bi-encoder models using sentence-transformers or custom backends."""

    def __init__(
        self,
        model_name: str,
        device: str = None,
        task: Optional[str] = None,
        allow_gte: bool = False,
        use_fp16: bool = False,
        max_seq_length: Optional[int] = None,
    ):
        """Initialize bi-encoder.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda/cpu), auto-detected if None
            task: Optional task name (useful for Jina models)
            allow_gte: If True, enables GTE custom backend. Off by default because
                GTE caused runtime instability in prior experiments.
            use_fp16: If True, use float16 precision to reduce memory usage
            max_seq_length: Maximum sequence length for truncation (None = model default)
        """
        print(f"Loading bi-encoder: {model_name}")
        self.model_name = model_name
        self.task = task
        self.use_fp16 = use_fp16
        self.max_seq_length = max_seq_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_fp16:
            print(f"⚠️  Using FP16 precision for memory efficiency")
        if max_seq_length:
            print(f"⚠️  Truncating sequences to max_seq_length={max_seq_length}")

        name_lower = model_name.lower()

        self.backend = "sentence_transformers"
        self.model = None

        # -------- INSTRUCTOR backend --------
        if "instructor" in name_lower:
            self.backend = "instructor"
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True,
            )
            print(f"INSTRUCTOR model loaded on device: {self.model.device}")

        # -------- Jina backend --------
        elif "jina" in name_lower:
            self.backend = "jina"
            # Jina v3/v5 requires trust_remote_code for custom modules
            self.task = task or "text-matching"
            
            # Some Jina v5 models (like text-small) require default_task
            # Others (like text-small-text-matching) use different architecture and don't
            model_kwargs = {'trust_remote_code': True}
            if "text-matching" not in model_name:
                # Only add default_task if not already in model name
                model_kwargs['default_task'] = self.task
            
            # Add FP16 support for memory efficiency
            if use_fp16:
                model_kwargs['torch_dtype'] = torch.float16
            
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            
            fp16_msg = " (FP16)" if use_fp16 else ""
            print(f"Jina model loaded on device: {self.model.device} with task: {self.task}{fp16_msg}")

        # -------- GTE backend (optional) --------
        elif "gte" in name_lower:
            if not allow_gte:
                raise ValueError(
                    f"GTE model '{model_name}' is disabled by default because it "
                    "previously caused runtime instability. If you really want to "
                    "try it again, initialize BiEncoder(..., allow_gte=True)."
                )

            self.backend = "gte"
            self.model = _TransformersEncoder(
                model_name=model_name,
                device=self.device,
                trust_remote_code=True,
                pooling="cls",
                max_length=256,
            )

        # -------- Standard sentence-transformers backend --------
        else:
            self.backend = "sentence_transformers"
            
            # Add FP16 support for memory efficiency
            model_kwargs = {'trust_remote_code': True}
            if use_fp16:
                model_kwargs['torch_dtype'] = torch.float16
            
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
            )
            
            # Set max_seq_length if specified
            if max_seq_length:
                self.model.max_seq_length = max_seq_length
            
            fp16_msg = " (FP16)" if use_fp16 else ""
            print(f"Model loaded on device: {self.model.device}{fp16_msg}")

    def _build_instructor_inputs(
        self,
        texts: Union[str, List[str]],
        text_type: str = "text",
    ):
        """Build INSTRUCTOR-format inputs: [instruction, text]."""
        if isinstance(texts, str):
            texts = [texts]

        if text_type == "label":
            instruction = "Represent the label description for zero-shot classification:"
        else:
            instruction = "Represent the text for zero-shot classification:"

        return [[instruction, t] for t in texts]

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
        text_type: str = "text",
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar
            text_type: "text" or "label" (used by INSTRUCTOR and can be useful later)

        Returns:
            Numpy array of embeddings
        """
        # -------- INSTRUCTOR --------
        if self.backend == "instructor":
            inputs = self._build_instructor_inputs(texts, text_type=text_type)
            embeddings = self.model.encode(
                inputs,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embeddings

        # -------- Jina --------
        if self.backend == "jina":
            # Jina v2 doesn't require task parameter during encoding
            # It works directly with sentence-transformers interface
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            
            # Clear GPU cache after encoding to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embeddings

        # -------- GTE custom backends --------
        if self.backend in {"gte", "gte"}:
            return self.model.encode(
                texts=texts,
                batch_size=batch_size,
                normalize=normalize,
                show_progress=show_progress,
                text_type=text_type,
            )

        # -------- Standard sentence-transformers --------
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        # Clear GPU cache after encoding to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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