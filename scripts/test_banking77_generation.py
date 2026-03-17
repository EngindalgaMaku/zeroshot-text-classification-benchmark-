"""Test banking77 label generation with first 3 labels."""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Import the task-aware generator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_descriptions.task_aware_label_generator import (
    TaskAwareLabelGenerator,
    TemplateStore,
    DatasetTaskTypeConfig,
)
from scripts.generate_label_descriptions import DescriptionGenerator

# Test labels
TEST_LABELS = [
    "activate my card",
    "age limit", 
    "apple pay or google pay"
]

def main():
    print("="*70)
    print("Testing banking77 label generation (first 3 labels)")
    print("="*70)
    
    # Initialize
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    
    llm_generator = DescriptionGenerator(model)
    
    template_store = TemplateStore.load("src/label_descriptions/prompt_templates.yaml")
    dataset_config = DatasetTaskTypeConfig.load("src/label_descriptions/dataset_task_types.yaml")
    
    task_aware = TaskAwareLabelGenerator(
        template_store=template_store,
        dataset_config=dataset_config,
        llm_generator=llm_generator
    )
    
    # Test each label
    for i, label_text in enumerate(TEST_LABELS):
        print(f"\n{'='*70}")
        print(f"Label {i}: {label_text}")
        print('='*70)
        
        try:
            l2, l3_list = task_aware.generate_for_label(
                dataset_id="banking77",
                label_text=label_text
            )
            
            print(f"✓ Generation successful")
            print(f"\nL2: {l2}")
            print(f"\nL3:")
            for j, l3 in enumerate(l3_list, 1):
                print(f"  {j}. {l3}")
            
            # Validation passed if we got here
            print(f"\n✓ Validation passed")
            
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return False
    
    print(f"\n{'='*70}")
    print("✓ All 3 labels generated successfully!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
