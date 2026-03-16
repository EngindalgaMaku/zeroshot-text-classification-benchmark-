"""Test different prompt versions for AG News label generation.

This script generates descriptions using different prompts and compares results.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# AG News labels
AG_NEWS_LABELS = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Sci/Tech"
}

# BASELINE (Manual - Good Performance ~64%)
BASELINE = {
    0: "This text is about international events, global politics, diplomacy, conflicts, or world affairs.",
    1: "This text is about sports, matches, teams, athletes, tournaments, or competitions.",
    2: "This text is about business, markets, finance, companies, trade, or the economy.",
    3: "This text is about science, technology, computers, innovation, research, or digital products."
}

# CURRENT LLM (Poor Performance ~55%)
PROMPT_V1_CURRENT = """Define the following text classification label in 15–20 words.

Focus on the semantic meaning of the label.
Do NOT repeat the label word itself.
Write a neutral description that could help classify a text belonging to this category.

Dataset: {dataset_name}
Label: {label_name}"""

# OPTIMIZED V2 (Domain + Concrete)
PROMPT_V2_DOMAIN_CONCRETE = """You are creating a text classification description for news articles.

Write EXACTLY ONE sentence that starts with "This text is about" and contains 10-15 words.

Requirements:
- Use CONCRETE, specific words that appear in actual {dataset_name} texts
- Use unique trigger words that DON'T appear in other class descriptions
- Avoid abstract academic language (e.g., "geopolitical", "broader culture")
- Avoid generic words that apply to multiple classes (e.g., "global", "developments", "impact")
- Focus on SPECIFIC nouns and verbs (e.g., "matches", "teams", "athletes" for sports)

Dataset: {dataset_name}
Label: {label_name}

Example format: "This text is about sports matches, teams, athletes, and tournament results."

Description:"""

# ULTRA-SPECIFIC V3 (Maximum Distinction)
PROMPT_V3_ULTRA_SPECIFIC = """Generate a classification description for news category: {label_name}

STRICT FORMAT: "This text is about [X, Y, Z]" (exactly 10-12 words)

RULES:
1. List 3-5 CONCRETE nouns/terms from actual {dataset_name} articles
2. Each word must be UNIQUE to this category (not used in other categories)
3. Use simple, everyday vocabulary (not academic jargon)
4. No abstract concepts, only tangible things

AVOID these generic words entirely:
- global, international, major, significant, important
- developments, impacts, issues, aspects, factors
- various, different, multiple, several, many

GOOD: "This text is about football matches, player transfers, league tables, and tournament scores."
BAD: "This text is about various sporting activities and competitive events globally."

Label: {label_name}
Description:"""

# COMPARISON V4 (Inspired by Manual)
PROMPT_V4_MANUAL_INSPIRED = """Create a classification description that matches this style:

EXAMPLE STYLE:
- "This text is about sports, matches, teams, athletes, tournaments, or competitions."
- "This text is about business, markets, finance, companies, trade, or the economy."

Your task: Create a similar description for "{label_name}" in {dataset_name} dataset.

REQUIREMENTS:
- Start with: "This text is about"
- List 6-8 concrete keywords separated by commas
- End with "or [category-general-term]"
- Use only simple, common words
- Total length: 10-15 words

Label: {label_name}
Description:"""


def generate_with_prompt(prompt_template: str, dataset_name: str = "ag_news") -> dict:
    """Generate descriptions using a specific prompt."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Generating with: {prompt_template.split('\n')[0][:50]}...")
    print('='*70)
    
    for label_id, label_name in AG_NEWS_LABELS.items():
        prompt = prompt_template.format(
            dataset_name=dataset_name,
            label_name=label_name
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        
        description = response.choices[0].message.content.strip()
        results[label_id] = description
        
        print(f"\n{label_id}: {label_name}")
        print(f"   → {description}")
    
    return results


def save_results(results: dict, version: str, output_dir: Path):
    """Save generated descriptions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ag_news_descriptions_{version}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_file}")


def main():
    """Generate descriptions with all prompt versions."""
    output_dir = Path("scripts/prompt_engineering/results")
    
    # Baseline (manual)
    print("\n" + "="*70)
    print("BASELINE (Manual - Good Performance)")
    print("="*70)
    for label_id, desc in BASELINE.items():
        print(f"\n{label_id}: {AG_NEWS_LABELS[label_id]}")
        print(f"   → {desc}")
    save_results(BASELINE, "baseline_manual", output_dir)
    
    # V1: Current (poor)
    # print("\n\nGenerating V1 (Current - Poor)...")
    # v1_results = generate_with_prompt(PROMPT_V1_CURRENT)
    # save_results(v1_results, "v1_current", output_dir)
    
    # V2: Domain + Concrete
    print("\n\nGenerating V2 (Domain + Concrete)...")
    v2_results = generate_with_prompt(PROMPT_V2_DOMAIN_CONCRETE)
    save_results(v2_results, "v2_domain_concrete", output_dir)
    
    # V3: Ultra-specific
    print("\n\nGenerating V3 (Ultra-Specific)...")
    v3_results = generate_with_prompt(PROMPT_V3_ULTRA_SPECIFIC)
    save_results(v3_results, "v3_ultra_specific", output_dir)
    
    # V4: Manual-inspired
    print("\n\nGenerating V4 (Manual-Inspired)...")
    v4_results = generate_with_prompt(PROMPT_V4_MANUAL_INSPIRED)
    save_results(v4_results, "v4_manual_inspired", output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL VERSIONS GENERATED!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}")
    print("\nNext steps:")
    print("1. Review generated descriptions")
    print("2. Create test configs for INSTRUCTOR")
    print("3. Run experiments and compare results")


if __name__ == "__main__":
    main()