"""Update MULTI_DATASET_EXPERIMENTS notebook to include GoEmotions"""
import json

notebook_path = "notebooks/MULTI_DATASET_EXPERIMENTS.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cells
for cell in nb['cells']:
    if 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Update dataset list check
        if '"ag_news", "dbpedia", "yahoo", "banking", "twitter", "financial", "20_newsgroups", "SetFit"' in source:
            new_source = source.replace(
                '"ag_news", "dbpedia", "yahoo", "banking", "twitter", "financial", "20_newsgroups", "SetFit"',
                '"ag_news", "dbpedia", "yahoo", "banking", "twitter", "financial", "20_newsgroups", "SetFit", "go_emotions", "goemotions"'
            )
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            print("✅ Updated dataset list check")
        
        # Add GoEmotions to dataset mapping
        if 'dataset = "20 Newsgroups"' in source and 'GoEmotions' not in source:
            new_source = source.replace(
                '        elif "20_newsgroups" in exp_name or "SetFit" in exp_name:\n            dataset = "20 Newsgroups"\n        else:\n            continue',
                '        elif "20_newsgroups" in exp_name or "SetFit" in exp_name:\n            dataset = "20 Newsgroups"\n        elif "go_emotions" in exp_name or "goemotions" in exp_name:\n            dataset = "GoEmotions"\n        else:\n            continue'
            )
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            print("✅ Added GoEmotions to dataset mapping")
        
        # Update dataset order in visualizations
        if 'dataset_order = [' in source and 'GoEmotions' not in source:
            new_source = source.replace(
                'dataset_order = ["AG News", "DBpedia-14", "Yahoo Answers", "Banking77", "Twitter Financial", "20 Newsgroups"]',
                'dataset_order = ["AG News", "DBpedia-14", "Yahoo Answers", "Banking77", "Twitter Financial", "20 Newsgroups", "GoEmotions"]'
            )
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            print("✅ Updated dataset order in visualizations")
        
        # Update model order
        if 'model_order = [' in source and '"Snowflake Arctic"' not in source:
            new_source = source.replace(
                'model_order = ["INSTRUCTOR", "Qwen3", "E5-large", "MPNet", "Jina v5", "BGE-M3"]',
                'model_order = ["INSTRUCTOR", "Snowflake Arctic", "Qwen3", "E5-large", "MPNet", "Jina v5", "BGE-M3"]'
            )
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            print("✅ Updated model order")

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook updated: {notebook_path}")
