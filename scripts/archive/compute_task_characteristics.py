"""
Compute task characteristics for each dataset.

This script computes:
- Number of classes for each dataset (from existing results)
- Average text length for each dataset
- Label semantic similarity scores using sentence embeddings

**Validates: Requirements 8.1, 8.2, 8.3**
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Try to import sentence_transformers (requires PyTorch)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Note: sentence_transformers not available (PyTorch missing). Using TF-IDF for label similarity.")

# Dataset configurations for loading text data
DATASET_CONFIGS = {
    "20 Newsgroups": {
        "hf_name": "SetFit/20_newsgroups",
        "split": "test",
        "text_col": "text",
        "max_samples": 2000
    },
    "AG News": {
        "hf_name": "ag_news",
        "split": "test",
        "text_col": "text",
        "max_samples": 1000
    },
    "Banking77": {
        "hf_name": "legacy-datasets/banking77",
        "split": "test",
        "text_col": "text",
        "max_samples": 1000
    },
    "DBpedia-14": {
        "hf_name": "dbpedia_14",
        "split": "test",
        "text_col": "content",
        "max_samples": 1000
    },
    "GoEmotions": {
        "hf_name": "google-research-datasets/go_emotions",
        "subset": "simplified",
        "split": "test",
        "text_col": "text",
        "max_samples": 1000
    },
    "IMDB": {
        "hf_name": "imdb",
        "split": "test",
        "text_col": "text",
        "max_samples": 1000
    },
    "SST-2": {
        "hf_name": "glue",
        "subset": "sst2",
        "split": "validation",
        "text_col": "sentence",
        "max_samples": 1000
    },
    "Yahoo Answers": {
        "hf_name": "yahoo_answers_topics",
        "split": "test",
        "text_col": "question_title",
        "max_samples": 1000
    },
    "Twitter Financial": {
        "hf_name": "zeroshot/twitter-financial-news-sentiment",
        "split": "validation",
        "text_col": "text",
        "max_samples": 1000
    }
}

# Label names for each dataset
LABEL_NAMES = {
    "20 Newsgroups": [
        "atheism", "graphics", "microsoft windows", "ibm hardware", "mac hardware",
        "x windows", "for sale", "autos", "motorcycles", "baseball", "hockey",
        "cryptography", "electronics", "medicine", "space", "christianity",
        "guns", "middle east", "politics", "religion"
    ],
    "AG News": ["world", "sports", "business", "science and technology"],
    "Banking77": [
        "activate my card", "age limit", "apple pay or google pay", "atm support",
        "automatic top up", "balance not updated after bank transfer", "balance not updated after cheque or cash deposit",
        "beneficiary not allowed", "cancel transfer", "card about to expire", "card acceptance",
        "card arrival", "card delivery estimate", "card linking", "card not working",
        "card payment fee charged", "card payment not recognised", "card payment wrong exchange rate",
        "card swallowed", "cash withdrawal charge", "cash withdrawal not recognised", "change pin",
        "compromised card", "contactless not working", "country support", "declined card payment",
        "declined cash withdrawal", "declined transfer", "direct debit payment not recognised",
        "disposable card limits", "edit personal details", "exchange charge", "exchange rate",
        "exchange via app", "extra charge on statement", "failed transfer", "fiat currency support",
        "get disposable virtual card", "get physical card", "getting spare card", "getting virtual card",
        "lost or stolen card", "lost or stolen phone", "order physical card", "passcode forgotten",
        "pending card payment", "pending cash withdrawal", "pending top up", "pending transfer",
        "pin blocked", "receiving money", "refund not showing up", "request refund", "reverted card payment",
        "supported cards and currencies", "terminate account", "top up by bank transfer charge",
        "top up by card charge", "top up by cash or cheque", "top up failed", "top up limits",
        "top up reverted", "topping up by card", "transaction charged twice", "transfer fee charged",
        "transfer into account", "transfer not received by recipient", "transfer timing", "unable to verify identity",
        "verify my identity", "verify source of funds", "verify top up", "virtual card not working",
        "visa or mastercard", "why verify identity", "wrong amount of cash received", "wrong exchange rate for cash withdrawal"
    ],
    "DBpedia-14": [
        "company", "educational institution", "artist", "athlete", "office holder",
        "mean of transportation", "building", "natural place", "village", "animal",
        "plant", "album", "film", "written work"
    ],
    "GoEmotions": [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ],
    "IMDB": ["negative", "positive"],
    "SST-2": ["negative", "positive"],
    "Yahoo Answers": [
        "society and culture", "science and mathematics", "health", "education and reference",
        "computers and internet", "sports", "business and finance", "entertainment and music",
        "family and relationships", "politics and government"
    ],
    "Twitter Financial": ["negative", "positive", "neutral"]
}


def load_dataset_texts(dataset_name, config):
    """Load dataset and return texts."""
    print(f"Loading {dataset_name}...")
    
    try:
        if "subset" in config:
            dataset = load_dataset(config["hf_name"], config["subset"], split=config["split"])
        else:
            dataset = load_dataset(config["hf_name"], split=config["split"])
    except Exception as e:
        print(f"  Warning: Could not load dataset: {e}")
        print(f"  Skipping {dataset_name}")
        return None
    
    # Sample if needed
    max_samples = config.get("max_samples")
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    texts = dataset[config["text_col"]]
    return texts


def compute_num_classes(dataset_name):
    """Compute number of classes for a dataset."""
    return len(LABEL_NAMES[dataset_name])


def compute_avg_text_length(texts):
    """Compute average text length in characters."""
    lengths = [len(text) for text in texts]
    return np.mean(lengths)


def compute_label_similarity(label_names, model_name="all-MiniLM-L6-v2"):
    """
    Compute label semantic similarity using sentence embeddings or TF-IDF fallback.
    Returns the average pairwise cosine similarity between label embeddings.
    
    Uses sentence_transformers if available (requires PyTorch), otherwise falls back
    to TF-IDF character n-gram vectors which capture lexical similarity.
    """
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"  Computing label similarity using {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(label_names, convert_to_numpy=True)
    else:
        print(f"  Computing label similarity using TF-IDF (PyTorch not available)...")
        # Use character n-gram TF-IDF as a lexical similarity proxy
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        embeddings = vectorizer.fit_transform(label_names).toarray()
        # Handle case where all labels are identical (zero vectors)
        if embeddings.sum() == 0:
            return 0.0

    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Get upper triangle (excluding diagonal) to avoid counting each pair twice
    n = len(label_names)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_similarities = similarities[upper_triangle_indices]

    # Return average similarity
    avg_similarity = float(np.mean(pairwise_similarities))
    return avg_similarity


def main():
    """Compute task characteristics for all datasets."""
    print("Computing task characteristics for all datasets...")
    print("=" * 80)
    
    results = []
    
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Compute number of classes
        num_classes = compute_num_classes(dataset_name)
        print(f"  Number of classes: {num_classes}")
        
        # Load texts and compute average length
        texts = load_dataset_texts(dataset_name, config)
        if texts is None:
            print(f"  Skipping {dataset_name} due to loading error")
            continue
            
        avg_text_length = compute_avg_text_length(texts)
        print(f"  Average text length: {avg_text_length:.2f} characters")
        
        # Compute label similarity
        label_names = LABEL_NAMES[dataset_name]
        label_similarity = compute_label_similarity(label_names)
        print(f"  Label similarity: {label_similarity:.4f}")
        
        results.append({
            "dataset": dataset_name,
            "num_classes": num_classes,
            "avg_text_length": avg_text_length,
            "label_similarity": label_similarity
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs("results/task_characteristics", exist_ok=True)
    
    output_path = "results/task_characteristics/task_characteristics.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"Task characteristics saved to {output_path}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
