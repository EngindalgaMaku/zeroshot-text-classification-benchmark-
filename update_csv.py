import json
import csv

# Read existing CSV
with open('results/MULTI_DATASET_RESULTS.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    existing = list(reader)
    fieldnames = reader.fieldnames

print(f"Existing rows: {len(existing)}")

# Load new experiments
files = [
    'results/raw/exp_SetFit_20_newsgroups_qwen3_metrics.json',
    'results/raw/exp_yahoo_answers_topics_qwen3_metrics.json'
]

# Dataset name mapping
dataset_map = {
    'SetFit/20_newsgroups': '20 Newsgroups',
    'yahoo_answers_topics': 'Yahoo Answers'
}

# Number of classes per dataset
num_classes_map = {
    'SetFit/20_newsgroups': 20,
    'yahoo_answers_topics': 10
}

new_rows = []
for f in files:
    with open(f, 'r') as mf:
        data = json.load(mf)
        dataset_name = dataset_map.get(data['dataset'], data['dataset'])
        num_classes = num_classes_map.get(data['dataset'], 20)
        new_rows.append({
            'dataset': dataset_name,
            'model': 'Qwen3',
            'samples': data['num_samples'],
            'num_classes': num_classes,
            'accuracy': data['accuracy'] * 100,
            'macro_f1': data['macro_f1'] * 100,
        })

print(f"Adding {len(new_rows)} new rows")
for row in new_rows:
    print(f"  {row['dataset']} - Qwen3: F1={row['macro_f1']:.3f}")

# Write updated CSV
with open('results/MULTI_DATASET_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing + new_rows)

print("✅ CSV updated!")