"""Quick test: Compare Manual vs V4 LLM descriptions on AG News with INSTRUCTOR.

This will run 3 experiments:
1. L1 (name_only) - baseline
2. Manual descriptions - good performance baseline
3. V4 descriptions - new LLM-generated
"""

import json
import yaml
import subprocess
from pathlib import Path

# Load V4 descriptions
v4_file = Path("scripts/prompt_engineering/results/ag_news_descriptions_v4_manual_inspired.json")
with open(v4_file) as f:
    v4_descriptions = json.load(f)

# Create temporary label descriptions for V4
temp_descriptions = {
    "ag_news": {
        str(k): {"l2": v} for k, v in v4_descriptions.items()
    }
}

# Save temporary descriptions file
temp_desc_file = Path("scripts/prompt_engineering/temp_v4_descriptions.json")
with open(temp_desc_file, 'w') as f:
    json.dump(temp_descriptions, f, indent=2)

# Create 3 test configs
configs = []

# Config 1: L1 (name_only)
config_l1 = {
    "experiment_name": "ag_news_instructor_l1_test",
    "dataset": {
        "name": "ag_news",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000
    },
    "task": {
        "label_mode": "name_only"
    },
    "models": {
        "biencoder": {
            "name": "hkunlp/instructor-base"
        }
    },
    "pipeline": {
        "normalize_embeddings": True,
        "random_seed": 42
    },
    "output": {
        "output_dir": "scripts/prompt_engineering/results",
        "save_metrics": True,
        "save_predictions": True
    }
}

# Config 2: Manual descriptions
config_manual = {
    "experiment_name": "ag_news_instructor_manual_test",
    "dataset": {
        "name": "ag_news",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000
    },
    "task": {
        "label_mode": "description"  # Uses manual descriptions from labels.py
    },
    "models": {
        "biencoder": {
            "name": "hkunlp/instructor-base"
        }
    },
    "pipeline": {
        "normalize_embeddings": True,
        "random_seed": 42
    },
    "output": {
        "output_dir": "scripts/prompt_engineering/results",
        "save_metrics": True,
        "save_predictions": True
    }
}

# Config 3: V4 descriptions  
config_v4 = {
    "experiment_name": "ag_news_instructor_v4_test",
    "dataset": {
        "name": "ag_news",
        "split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000
    },
    "task": {
        "label_mode": "l2"  # Will use temp_v4_descriptions.json
    },
    "models": {
        "biencoder": {
            "name": "hkunlp/instructor-base"
        }
    },
    "pipeline": {
        "normalize_embeddings": True,
        "random_seed": 42
    },
    "output": {
        "output_dir": "scripts/prompt_engineering/results",
        "save_metrics": True,
        "save_predictions": True
    }
}

# Save configs
config_dir = Path("scripts/prompt_engineering/configs")
config_dir.mkdir(parents=True, exist_ok=True)

configs_info = [
    (config_l1, "l1_name_only"),
    (config_manual, "manual_descriptions"),
    (config_v4, "v4_llm_descriptions")
]

print("="*70)
print("CREATING TEST CONFIGS")
print("="*70)

for config, name in configs_info:
    config_file = config_dir / f"ag_news_instructor_{name}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Created: {config_file}")
    configs.append(config_file)

print("\n" + "="*70)
print("READY TO RUN EXPERIMENTS")
print("="*70)
print("\nTo run tests:")
print(f"python main.py {configs[0]}")
print(f"python main.py {configs[1]}")
print(f"python main.py {configs[2]}")
print("\nOr run all at once:")
for cfg in configs:
    print(f"python main.py {cfg}")

print("\n" + "="*70)
print("NOTE: Make sure to temporarily replace generated_descriptions.json")
print(f"with: {temp_desc_file}")
print("Or update src/labels.py to load from temp file for testing")
print("="*70)