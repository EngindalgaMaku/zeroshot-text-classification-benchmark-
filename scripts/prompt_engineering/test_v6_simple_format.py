"""V6: Exact manual format replication with LLM content."""

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

# V6: EXACT MANUAL FORMAT
PROMPT_V6_SIMPLE = """You are generating a text classification description for news articles.

Dataset: AG News (news articles from 4 categories)
Label: {label_name}

STRICT FORMAT - Output EXACTLY this pattern:
"This text is about [term1], [term2], [term3], [term4], [term5], or [term6]."

CRITICAL RULES:
1. Start with EXACTLY: "This text is about"
2. List 6-8 concrete, domain-specific terms
3. Separate with commas: ", "
4. Last term uses "or" instead of comma
5. End with period
6. NO other sentences, NO explanations, JUST this one sentence!

CONTENT RULES:
- Use journalism/news domain terminology
- Use concrete nouns (NOT abstract concepts)
- Each term must be specific to {label_name} news
- NO generic words (events, issues, topics, things, aspects)

GOOD EXAMPLE:
"This text is about wars, treaties, diplomacy, leaders, sanctions, or foreign policy."

BAD EXAMPLES:
- "This is about various events..." (too generic)
- "News covering..." (wrong start)
- "Keywords: wars, treaties" (structured format)

Now generate ONLY the single sentence for:
Label: {label_name}

Output:"""

def generate_v6(label_name: str) -> str:
    """Generate single V6 description."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    
    prompt = PROMPT_V6_SIMPLE.format(label_name=label_name)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80,
    )
    
    description = response.choices[0].message.content.strip()
    
    # Validate format
    if not description.startswith("This text is about"):
        print(f"⚠️  WARNING: Incorrect format for {label_name}")
        print(f"   Got: {description}")
    
    return description


def main():
    """Generate V6 simple format descriptions."""
    print("="*70)
    print("V6: SIMPLE FORMAT (Manuel Replica)")
    print("="*70)
    
    v6_results = {}
    
    for label_id, label_name in AG_NEWS_LABELS.items():
        print(f"\n{label_id}: {label_name}")
        description = generate_v6(label_name)
        v6_results[label_id] = description
        print(f"   → {description}")
    
    # Save
    output_dir = Path("scripts/prompt_engineering/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple JSON
    output_file = output_dir / "ag_news_descriptions_v6_simple.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(v6_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {output_file}")
    
    # L2 format for testing
    v6_l2_format = {
        "ag_news": {
            str(k): {"l2": v}
            for k, v in v6_results.items()
        }
    }
    
    l2_file = output_dir / "ag_news_descriptions_v6_l2_format.json"
    with open(l2_file, 'w', encoding='utf-8') as f:
        json.dump(v6_l2_format, f, indent=2, ensure_ascii=False)
    print(f"✅ L2 format: {l2_file}")
    
    print("\n" + "="*70)
    print("✅ V6 SIMPLE FORMAT GENERATED!")
    print("="*70)
    print("\nFormat matches manual:")
    print('  "This text is about X, Y, Z, W, or V."')
    print("\nNext:")
    print("1. Copy to src/label_descriptions/")
    print("2. Test with INSTRUCTOR")
    print("\nTarget: ≥82% F1 (close to manual 84.2%)")


if __name__ == "__main__":
    main()