"""Test parsing logic for experiment config filenames."""

from pathlib import Path

def parse_filename(stem):
    """Parse filename: full_{dataset}_{model}_{mode}.yaml"""
    if not stem.startswith("full_"):
        return None, None, None
    
    rest = stem[5:]  # Remove 'full_'
    
    # Check for known modes at the end
    if rest.endswith("_name_only"):
        mode = "name_only"
        rest = rest[:-10]
    elif rest.endswith("_l2"):
        mode = "l2"
        rest = rest[:-3]
    elif rest.endswith("_l3"):
        mode = "l3"
        rest = rest[:-3]
    else:
        return None, None, None
    
    # Split dataset and model
    parts = rest.rsplit('_', 1)
    if len(parts) != 2:
        return None, None, None
    
    dataset, model = parts[0], parts[1]
    return dataset, model, mode

# Test cases
test_cases = [
    "full_SetFit_20_newsgroups_bge_name_only",
    "full_SetFit_20_newsgroups_bge_l2",
    "full_SetFit_20_newsgroups_bge_l3",
    "full_ag_news_mpnet_name_only",
    "full_ag_news_mpnet_l2",
    "full_yahoo_answers_topics_e5_l3",
    "full_zeroshot_twitter_financial_news_sentiment_instructor_name_only",
]

print("Testing filename parsing:")
print("=" * 60)
for stem in test_cases:
    dataset, model, mode = parse_filename(stem)
    print(f"{stem}")
    print(f"  -> dataset: {dataset}")
    print(f"  -> model: {model}")
    print(f"  -> mode: {mode}")
    print()

# Test grouping
print("\nTesting grouping:")
print("=" * 60)
configs_by_group = {}
for stem in test_cases:
    dataset, model, mode = parse_filename(stem)
    if dataset and model and mode:
        base_name = f"{dataset}_{model}"
        if base_name not in configs_by_group:
            configs_by_group[base_name] = {}
        configs_by_group[base_name][mode] = stem

for name, modes in sorted(configs_by_group.items()):
    print(f"{name}: {sorted(modes.keys())}")

print(f"\nTotal groups: {len(configs_by_group)}")
print(f"Total experiments: {sum(len(m) for m in configs_by_group.values())}")
