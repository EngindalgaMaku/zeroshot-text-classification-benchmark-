"""Complete update of MULTI_DATASET_EXPERIMENTS notebook for GoEmotions"""
import json
import re

notebook_path = "notebooks/MULTI_DATASET_EXPERIMENTS.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updates = 0

# Update all cells
for cell in nb['cells']:
    if 'source' in cell:
        source_list = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
        source = ''.join(source_list)
        original = source
        
        # Update "6 datasets" to "7 datasets" (case insensitive)
        source = re.sub(r'\b6 datasets\b', '7 datasets', source, flags=re.IGNORECASE)
        source = re.sub(r'\b6 DATASETS\b', '7 DATASETS', source)
        
        # Update "42 experiments" to "49 experiments"
        source = source.replace('42 experiments', '49 experiments')
        source = source.replace('42 Experiments', '49 Experiments')
        
        # Update dataset list in checks
        source = source.replace(
            '"ag_news", "dbpedia", "yahoo", "banking", "twitter", "financial", "20_newsgroups", "SetFit"',
            '"ag_news", "dbpedia", "yahoo", "banking", "twitter", "financial", "20_newsgroups", "SetFit", "go_emotions", "goemotions"'
        )
        
        # Add GoEmotions to dataset mapping (if not already there)
        if '"20 Newsgroups"' in source and 'GoEmotions' not in source and 'dataset = ' in source:
            source = source.replace(
                '        elif "20_newsgroups" in exp_name or "SetFit" in exp_name:\n            dataset = "20 Newsgroups"\n        else:\n            continue',
                '        elif "20_newsgroups" in exp_name or "SetFit" in exp_name:\n            dataset = "20 Newsgroups"\n        elif "go_emotions" in exp_name or "goemotions" in exp_name:\n            dataset = "GoEmotions"\n        else:\n            continue'
            )
        
        # Update difficulty ranking to include GoEmotions
        if 'Difficulty Ranking:' in source and 'GoEmotions' not in source:
            source = source.replace(
                '6. **Banking77** (77 classes) - VERY HARD!',
                '6. **Banking77** (77 classes) - VERY HARD!\n7. **GoEmotions** (28 classes) - HARD (fine-grained emotions)'
            )
        
        # Update "7 Models" references
        source = source.replace('ALL 7 Models', 'ALL 7 Models')
        source = source.replace('7 models', '7 models')
        
        # Update model count in titles
        source = source.replace('(7 Models)', '(7 Models)')
        
        if source != original:
            # Convert back to list format
            cell['source'] = source.split('\n') if '\n' in source else [source]
            updates += 1

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Updated {updates} cells")
print(f"✅ Notebook updated: {notebook_path}")
