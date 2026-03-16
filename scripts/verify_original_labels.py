"""Verify that our label names match the original dataset labels.

This script loads each dataset from HuggingFace and compares the actual
label names with what we have defined in src/labels.py
"""

from datasets import load_dataset
from src.labels import LABEL_SETS
import sys


def get_dataset_labels(dataset_name, split="train"):
    """Get original label names from HuggingFace dataset."""
    print(f"\n{'='*70}")
    print(f"Checking: {dataset_name}")
    print('='*70)
    
    try:
        # Load a small sample to get label info
        if dataset_name == "sst2":
            ds = load_dataset("glue", "sst2", split=f"{split}[:10]")
        else:
            ds = load_dataset(dataset_name, split=f"{split}[:10]", trust_remote_code=True)
        
        # Get label feature
        if "label" in ds.features:
            label_feature = ds.features["label"]
        elif "labels" in ds.features:
            label_feature = ds.features["labels"]
        else:
            print(f"⚠️  No 'label' or 'labels' column found")
            print(f"   Available columns: {list(ds.features.keys())}")
            return None
        
        # Try to get label names
        if hasattr(label_feature, "names"):
            # ClassLabel feature
            return {i: name for i, name in enumerate(label_feature.names)}
        elif hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
            # Sequence(ClassLabel) feature (for multi-label)
            return {i: name for i, name in enumerate(label_feature.feature.names)}
        else:
            print(f"⚠️  Cannot extract label names from feature type: {type(label_feature)}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def compare_labels(dataset_name, split="train"):
    """Compare our labels with original dataset labels."""
    
    # Get original labels from HuggingFace
    original = get_dataset_labels(dataset_name, split)
    
    if original is None:
        return False
    
    # Get our labels from LABEL_SETS
    if dataset_name not in LABEL_SETS:
        print(f"❌ Dataset not found in LABEL_SETS")
        return False
    
    our_labels = LABEL_SETS[dataset_name]["name_only"]
    
    # Compare
    all_match = True
    print(f"\n{'ID':<5} {'ORIGINAL':<40} {'OURS':<40} {'MATCH'}")
    print('-'*90)
    
    for label_id in sorted(set(list(original.keys()) + list(our_labels.keys()))):
        orig_name = original.get(label_id, "MISSING")
        our_name = our_labels.get(label_id, ["MISSING"])[0]
        
        # Normalize for comparison
        orig_normalized = orig_name.lower().replace("_", " ").replace("/", " ").strip()
        our_normalized = our_name.lower().replace("_", " ").replace("/", " ").strip()
        
        match = orig_normalized == our_normalized
        match_symbol = "✅" if match else "❌"
        
        if not match:
            all_match = False
        
        print(f"{label_id:<5} {orig_name:<40} {our_name:<40} {match_symbol}")
    
    print()
    if all_match:
        print("✅ ALL LABELS MATCH!")
    else:
        print("❌ SOME LABELS DON'T MATCH - NEEDS FIXING")
    
    return all_match


def main():
    """Check all datasets."""
    datasets_to_check = [
        ("ag_news", "test"),
        ("dbpedia_14", "test"),
        ("yahoo_answers_topics", "test"),
        ("banking77", "test"),
        ("zeroshot/twitter-financial-news-sentiment", "test"),
        ("SetFit/20_newsgroups", "test"),
        ("imdb", "test"),
        ("sst2", "validation"),  # SST-2 uses 'validation' split
        ("go_emotions", "test"),
    ]
    
    results = {}
    
    for dataset_name, split in datasets_to_check:
        match = compare_labels(dataset_name, split)
        results[dataset_name] = match
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for dataset_name, match in results.items():
        status = "✅ MATCH" if match else "❌ MISMATCH"
        print(f"{dataset_name:<50} {status}")
    
    total = len(results)
    matched = sum(1 for m in results.values() if m)
    
    print(f"\n{matched}/{total} datasets have matching labels")
    
    if matched < total:
        print("\n⚠️  Some datasets need label corrections in src/labels.py!")
        sys.exit(1)
    else:
        print("\n✅ All datasets verified!")
        sys.exit(0)


if __name__ == "__main__":
    main()