"""Test V5: Structured descriptions (Semantic + Keywords + Examples)."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Domain info
DATASET_INFO = {
    "name": "ag_news",
    "domain": "news articles",
    "description": "a dataset of news articles from various categories"
}

# V5_L2: Semantic + Keywords
PROMPT_V5_L2 = """Generate a structured text classification description for news classification.

Dataset: {dataset_name} ({dataset_domain})
Label: {label_name}

REQUIRED FORMAT (MUST FOLLOW EXACTLY):
"[One clear sentence defining this category]. Keywords: [4-6 domain-specific terms]"

CRITICAL RULES:
1. First sentence: Clear semantic meaning of the category
2. Keywords: ONLY domain-specific nouns/terms from {dataset_domain}
3. Keywords must be CONCRETE and UNIQUE to this category
4. NO generic everyday words (events, issues, topics, aspects, developments, etc.)
5. Use technical/professional vocabulary from journalism

BAD EXAMPLE (too generic):
"News about various global events. Keywords: topics, issues, developments, aspects, news"

GOOD EXAMPLE:
"News about international politics and diplomatic relations. Keywords: treaties, sanctions, summits, foreign policy, embassies"

Now generate for this category:
Label: {label_name}
Dataset: {dataset_name} ({dataset_domain})

Description:"""

# V5_L3: Semantic + Keywords + Examples  
PROMPT_V5_L3 = """Generate a comprehensive structured description for news classification.

Dataset: {dataset_name} ({dataset_domain})
Label: {label_name}

REQUIRED FORMAT (MUST FOLLOW EXACTLY):
"[One clear sentence defining this category]. Keywords: [4-6 domain-specific terms]. Examples: [2-3 typical content patterns]"

CRITICAL RULES:
1. First sentence: Clear semantic meaning
2. Keywords: Domain-specific NOUNS from {dataset_domain}
3. Examples: Typical article patterns/topics (NOT quotes!)
4. Everything must be CONCRETE and category-specific
5. NO generic words (events, issues, topics, various, major, etc.)

BAD EXAMPLE:
"News about various topics. Keywords: events, issues. Examples: important news, major developments"

GOOD EXAMPLE:
"News about international politics and diplomatic relations. Keywords: treaties, sanctions, summits, foreign policy, embassies. Examples: UN climate negotiations, NATO expansion talks, bilateral trade agreements"

Now generate for this category:
Label: {label_name}
Dataset: {dataset_name} ({dataset_domain})

Description:"""


def generate_descriptions(prompt_template: str, version: str) -> dict:
    """Generate descriptions using V5 structured prompts."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Generating {version}...")
    print('='*70)
    
    for label_id, label_name in AG_NEWS_LABELS.items():
        prompt = prompt_template.format(
            dataset_name=DATASET_INFO["name"],
            dataset_domain=DATASET_INFO["domain"],
            label_name=label_name
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        
        description = response.choices[0].message.content.strip()
        results[label_id] = description
        
        print(f"\n{label_id}: {label_name}")
        print(f"   → {description}")
    
    return results


def save_json(data: dict, version: str, output_dir: Path):
    """Save to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ag_news_descriptions_{version}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved: {output_file}")
    return output_file


def create_l2_l3_format(l2_data: dict, l3_data: dict, output_dir: Path):
    """Create L2+L3 combined format for testing."""
    combined = {
        "ag_news": {
            str(k): {
                "l2": l2_data[k],
                "l3": [l3_data[k]]  # Wrap in list
            }
            for k in l2_data.keys()
        }
    }
    
    output_file = output_dir / "ag_news_descriptions_v5_combined.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Combined L2+L3 saved: {output_file}")
    return output_file


def main():
    """Generate V5 structured descriptions."""
    output_dir = Path("scripts/prompt_engineering/results")
    
    # Generate V5_L2
    v5_l2 = generate_descriptions(PROMPT_V5_L2, "v5_l2_structured")
    save_json(v5_l2, "v5_l2_structured", output_dir)
    
    # Generate V5_L3
    v5_l3 = generate_descriptions(PROMPT_V5_L3, "v5_l3_structured")
    save_json(v5_l3, "v5_l3_structured", output_dir)
    
    # Create combined format
    create_l2_l3_format(v5_l2, v5_l3, output_dir)
    
    print("\n" + "="*70)
    print("✅ V5 STRUCTURED DESCRIPTIONS GENERATED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review descriptions")
    print("2. Copy combined file to src/label_descriptions/")
    print("3. Run tests with V5 prompts")
    print("\nExpected improvement:")
    print("- V5_L2 Target: ≥82% F1")
    print("- V5_L3 Target: ≥83% F1")
    print("- Manual: 84.2% F1 (gold standard)")


if __name__ == "__main__":
    main()