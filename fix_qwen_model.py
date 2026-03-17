"""Fix Qwen3 model name in all experiment configs."""

from pathlib import Path

config_dir = Path("src/label_descriptions/experiments")
fixed_count = 0

for config_file in config_dir.glob("*.yaml"):
    content = config_file.read_text(encoding="utf-8")
    
    # Check if it contains the wrong Qwen model name
    if "Qwen/Qwen3-Embedding" in content and "4B" not in content:
        # Fix the model name
        content = content.replace(
            "Qwen/Qwen3-Embedding",
            "Qwen/Qwen3-Embedding-4B"
        )
        config_file.write_text(content, encoding="utf-8")
        fixed_count += 1
        print(f"Fixed: {config_file.name}")

print(f"\nTotal fixed: {fixed_count}")
