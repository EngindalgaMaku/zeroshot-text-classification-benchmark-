"""Fix Qwen3 configs to 4B"""
import yaml
from pathlib import Path

print("="*70)
print("Fixing Qwen3 configs - 8B → 4B")
print("="*70)

qwen_fixed = 0
for config_file in Path("experiments").glob("*qwen3*.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    old_model = config['models']['biencoder']['name']
    
    # Fix to 4B
    config['models']['biencoder']['name'] = "Qwen/Qwen3-Embedding-4B"
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    qwen_fixed += 1
    print(f"✅ {config_file.name}: {old_model} → Qwen3-Embedding-4B")

print(f"\nFixed {qwen_fixed} Qwen3 configs")
print("="*70)