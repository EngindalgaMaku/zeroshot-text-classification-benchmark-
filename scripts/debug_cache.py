"""Debug script to check generated descriptions cache."""
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels import load_generated_descriptions, _GENERATED_DESC_CACHE

print("=== Checking generated_descriptions.json ===")
desc_file = Path(__file__).parent.parent / "src" / "label_descriptions" / "generated_descriptions.json"
print(f"File exists: {desc_file.exists()}")
print(f"File path: {desc_file}")

if desc_file.exists():
    with open(desc_file, encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\nDatasets in file: {list(data.keys())}")
    
    if "banking77" in data:
        print(f"Banking77 labels in file: {len(data['banking77'])}")
        # Check first label
        first_label = data['banking77']['0']
        print(f"\nFirst label structure:")
        print(f"  Keys: {list(first_label.keys())}")
        if 'l2' in first_label:
            print(f"  L2: {first_label['l2'][:80]}...")
        if 'l3' in first_label:
            print(f"  L3 count: {len(first_label['l3'])}")

print("\n=== Checking cache ===")
print(f"Cache before load: {_GENERATED_DESC_CACHE is not None}")

# Force reload
loaded = load_generated_descriptions(force_reload=True)
print(f"Cache after force reload: {_GENERATED_DESC_CACHE is not None}")

if "banking77" in loaded:
    print(f"Banking77 labels in cache: {len(loaded['banking77'])}")
    first_cached = loaded['banking77']['0']
    print(f"First cached label keys: {list(first_cached.keys())}")
else:
    print("ERROR: banking77 not in loaded cache!")
