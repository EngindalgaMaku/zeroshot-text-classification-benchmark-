import json

print("Reading notebook...")
with open('../notebooks/MULTI_DATASET_EXPERIMENTS.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed = False
for cell in nb['cells']:
    if 'source' in cell:
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check if this is the problematic cell
        if 'elif "imdb" in exp_name:      dataset' in src:
            print("Found problematic cell, fixing...")
            
            # Replace the entire problematic section
            old_text = 'elif "go_emotions" in exp_name or "goemotions" in exp_name:            dataset = "GoEmotions"       elif "imdb" in exp_name:      dataset = "IMDB"     elif "sst2" in exp_name:       dataset = "SST-2"'
            
            new_text = '''elif "go_emotions" in exp_name or "goemotions" in exp_name:
            dataset = "GoEmotions"
        elif "imdb" in exp_name:
            dataset = "IMDB"
        elif "sst2" in exp_name:
            dataset = "SST-2"'''
            
            src = src.replace(old_text, new_text)
            cell['source'] = src.split('\n')
            fixed = True
            print("Fixed indentation!")

if fixed:
    print("Saving notebook...")
    with open('../notebooks/MULTI_DATASET_EXPERIMENTS.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✅ Done! Notebook fixed successfully.")
else:
    print("⚠️ Could not find the problematic section. Maybe it's already fixed?")
