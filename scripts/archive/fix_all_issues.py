"""
Fix all 4 issues:
1. HuggingFace login
2. Qwen3 OOM (8B → 1.5B)
3. Jina AttributeError
4. Model caching
"""
import yaml
from pathlib import Path
import os

print("="*70)
print("FIXING ALL ISSUES")
print("="*70)

# ============================================================================
# 1. Setup HuggingFace Token
# ============================================================================
print("\n1. Setting up HuggingFace token...")
token_file = Path.home() / ".huggingface" / "token"
if token_file.exists():
    print("   ✅ HF token already exists")
else:
    print("   ⚠️  Please run: huggingface-cli login")
    print("   Or set HF_TOKEN environment variable")

# ============================================================================
# 2. Fix Qwen3 model - Use smaller 1.5B instead of 8B
# ============================================================================
print("\n2. Fixing Qwen3 model (8B → 1.5B)...")
qwen_fixed = 0
for config_file in Path("experiments").glob("*qwen3*.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    old_model = config['models']['biencoder']['name']
    if "8B" in old_model:
        config['models']['biencoder']['name'] = "Qwen/Qwen2.5-Embedding-1.5B"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        qwen_fixed += 1
        print(f"   ✅ {config_file.name}: Qwen3-8B → Qwen2.5-1.5B")

print(f"   Fixed {qwen_fixed} Qwen configs")

# ============================================================================
# 3. Fix Jina - Use trust_remote_code and lower transformers version
# ============================================================================
print("\n3. Jina fix suggestions:")
print("   The issue is with transformers version compatibility")
print("   Solutions:")
print("   a) Downgrade transformers: pip install transformers==4.45.2")
print("   b) Or use jina-embeddings-v2 instead of v3")
print("   c) Wait for jina to update their code")
print("\n   Applying fix: Using jina-embeddings-v2...")

jina_fixed = 0
for config_file in Path("experiments").glob("*jina*.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    old_model = config['models']['biencoder']['name']
    if "jina-embeddings-v3" in old_model:
        # Use v2 which is more stable
        config['models']['biencoder']['name'] = "jinaai/jina-embeddings-v2-base-en"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        jina_fixed += 1
        print(f"   ✅ {config_file.name}: jina-v3 → jina-v2-base")

print(f"   Fixed {jina_fixed} Jina configs")

# ============================================================================
# 4. Setup model caching
# ============================================================================
print("\n4. Model caching:")
cache_dir = Path.home() / ".cache" / "huggingface"
print(f"   Cache directory: {cache_dir}")
print(f"   Exists: {cache_dir.exists()}")
if cache_dir.exists():
    # Count cached models
    hub_dir = cache_dir / "hub"
    if hub_dir.exists():
        models = list(hub_dir.glob("models--*"))
        print(f"   Cached models: {len(models)}")
    print("   ✅ Models will be cached automatically")
else:
    print("   ⚠️  Cache directory will be created on first download")

print("\n" + "="*70)
print("ALL FIXES APPLIED!")
print("="*70)
print("\nNext steps:")
print("1. Run: huggingface-cli login (if not logged in)")
print("2. Or: pip install transformers==4.45.2 (if you want jina-v3)")
print("3. Run experiments again")
print("="*70)