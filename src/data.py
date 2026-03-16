"""Dataset loading and preprocessing."""

from datasets import load_dataset
from typing import Dict, Any, Optional
import pandas as pd


# Turkish dataset label mappings (string to int)
TURKISH_LABEL_MAPS = {
    "interpress_news_category_tr_lite": {
        "siyaset": 0,
        "dünya": 1,
        "ekonomi": 2,
        "kültür": 3,
        "sağlık": 4,
        "spor": 5,
        "teknoloji": 6,
    },
    "winvoker/turkish-sentiment-analysis-dataset": {
        "positive": 0,
        "negative": 1,
        "neutral": 2,
    },
    "savasy/ttc4900": {
        "siyaset": 0,
        "dünya": 1,
        "ekonomi": 2,
        "kültür": 3,
        "sağlık": 4,
        "spor": 5,
        "teknoloji": 6,
    },
}


def load_text_classification_dataset(cfg: Dict[str, Any], seed: Optional[int] = None):
    """Load a text classification dataset.
    
    Args:
        cfg: Configuration dictionary containing dataset parameters
        seed: Random seed for sampling (if None, tries to get from cfg["pipeline"]["random_seed"], defaults to 42)
        
    Returns:
        HuggingFace Dataset object
    """
    ds_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", "test")
    max_samples = cfg["dataset"].get("max_samples")
    
    # Get seed from parameter, config, or default
    if seed is None:
        seed = cfg.get("pipeline", {}).get("random_seed", 42)
    
    print(f"Loading dataset: {ds_name}, split: {split}")
    if max_samples:
        print(f"Random seed for sampling: {seed}")
    
    # Load dataset
    if ds_name == "ag_news":
        dataset = load_dataset("ag_news", split=split)
    elif ds_name == "dbpedia_14":
        dataset = load_dataset("dbpedia_14", split=split)
    elif ds_name == "yahoo_answers_topics":
        dataset = load_dataset("yahoo_answers_topics", split=split)
    elif ds_name == "imdb":
        dataset = load_dataset("imdb", split=split)
    elif ds_name == "sst2":
        # SST-2 is part of GLUE benchmark
        dataset = load_dataset("glue", "sst2", split=split)
    else:
        # Try to load as HuggingFace dataset
        # Some datasets require trust_remote_code for legacy scripts
        try:
            dataset = load_dataset(ds_name, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Failed with trust_remote_code=True, trying without: {e}")
            dataset = load_dataset(ds_name, split=split)
    
    # Sample if needed
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        print(f"Sampled {max_samples} examples (seed={seed})")
    
    print(f"Dataset loaded: {len(dataset)} examples")
    return dataset


def prepare_texts_and_labels(dataset, text_column: str, label_column: str, dataset_name: str = None):
    """Extract texts and labels from dataset.
    
    Args:
        dataset: HuggingFace Dataset
        text_column: Name of text column
        label_column: Name of label column
        dataset_name: Name of dataset (for Turkish label mapping and special handling)
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = dataset[text_column]
    labels = dataset[label_column]
    
    # Special handling for GoEmotions (multi-label dataset)
    if dataset_name == "go_emotions":
        print("⚠️  GoEmotions is multi-label - converting to single-label by taking first emotion")
        converted_labels = []
        for label_list in labels:
            if isinstance(label_list, (list, tuple)) and len(label_list) > 0:
                # Take the first label (most dominant emotion)
                converted_labels.append(label_list[0])
            elif isinstance(label_list, (list, tuple)) and len(label_list) == 0:
                # Empty list - assign neutral (label 27)
                converted_labels.append(27)
            else:
                # Already single label
                converted_labels.append(label_list)
        labels = converted_labels
        print(f"✅ Converted {len(labels)} multi-label examples to single-label")
    
    # Convert Turkish string labels to integers
    elif dataset_name and dataset_name in TURKISH_LABEL_MAPS:
        label_map = TURKISH_LABEL_MAPS[dataset_name]
        print(f"Converting string labels to integers using mapping: {label_map}")
        converted_labels = []
        for label in labels:
            if isinstance(label, str):
                if label not in label_map:
                    raise ValueError(f"Unknown label '{label}' not in mapping: {list(label_map.keys())}")
                converted_labels.append(label_map[label])
            else:
                converted_labels.append(label)
        labels = converted_labels
        print(f"✅ Converted {len(labels)} labels successfully")
    
    return texts, labels


def load_custom_dataset(file_path: str, text_column: str, label_column: str):
    """Load custom dataset from CSV or JSON.
    
    Args:
        file_path: Path to dataset file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (texts, labels)
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=file_path.endswith('.jsonl'))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    return texts, labels