#!/usr/bin/env python3
"""
Set ALL label formulation configs to batch_size=8
For consistency across all models and to avoid OOM with Qwen3
"""

import yaml
from pathlib import Path

# Find all label formulation configs
config_dir = Path("experiments/label_formulation")
all_configs = list(config_dir.glob("exp_*.yaml"))

print(f"Found {len(all_configs)} label formulation configs")
print(f"\n🔧 Setting ALL configs to batch_size=8 for consistency")
print(f"\nReason: Qwen3-8B needs batch_size=8 for 20 Newsgroups")
print(f"        → All models should use same batch_size for fair comparison\n")

updated_count = 0
already_correct = 0

for config_path in all_configs:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    current_batch = config.get('pipeline', {}).get('batch_size', 16)
    
    if current_batch != 8:
        config['pipeline']['batch_size'] = 8
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        model = config['models']['biencoder']['name'].split('/')[-1]
        dataset = config['dataset']['name'].split('/')[-1]
        label_mode = config['task']['label_mode']
        
        print(f"✅ {dataset[:15]:15} | {model[:20]:20} | {label_mode:11} | {current_batch} → 8")
        updated_count += 1
    else:
        already_correct += 1

print(f"\n{'='*70}")
print(f"Summary:")
print(f"  Updated: {updated_count} configs")
print(f"  Already correct: {already_correct} configs")
print(f"  Total: {len(all_configs)} configs")
print(f"\n✅ All label formulation configs now use batch_size=8")
print(f"\n💡 Benefits:")
print(f"   - Consistent across all models")
print(f"   - Works with Qwen3-8B on all datasets")
print(f"   - Fair comparison (same batch size for all)")