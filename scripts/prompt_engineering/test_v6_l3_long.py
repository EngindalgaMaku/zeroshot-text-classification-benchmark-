"""V6_L3: Long descriptions with elaboration."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# V6_L3: Extended format with elaboration
PROMPT_V6_L3 = """You are generating a comprehensive text classification description for news articles.

Dataset: AG News (news articles)
Label: {label_name}

OUTPUT 3 VARIATIONS - Each on a new line:

Line 1: "This text is about [6-8 domain terms separated by commas, last with 'or']."
Line 2: "[Elaborate on aspect 1 of the category in 8-12 words]"
Line 3: "[Elaborate on aspect 2 of the category in 8-12 words]"

CRITICAL RULES:
1. Line 1: Same format as short description
2. Lines 2-3: Expand on different aspects/contexts of the category
3. All 3 lines must use domain-specific concrete terms
4. NO generic words (events, issues, topics, various)
5. Each line should provide unique information

EXAMPLE for "World":
This text is about diplomacy, international relations, treaties, conflicts, humanitarian aid, or global summits.
Coverage includes breaking news from war zones, refugee crises, and peacekeeping missions.
Analysis focuses on geopolitical tensions, bilateral agreements, and United Nations actions.

EXAMPLE for "Sports":
This text is about athletes, championships, scores, leagues, tournaments, or records.
Coverage includes game highlights, player statistics, team rankings, and match results.
Analysis focuses on coaching strategies, player transfers, playoff races, and championship predictions.

Now generate 3 lines for:
Label: {label_name}

Output (3 lines):"""


def generate_v6_l3(label_name: str, client: OpenAI, model: str) -> list:
    """Generate V6 L3 description (3 variations)."""
    prompt = PROMPT_V6_L3.format(label_name=label_name)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    
    text = response.choices[0].message.content.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Ensure we have exactly 3 lines
    if len(lines) != 3:
        print(f"    ⚠️  Expected 3 lines, got {len(lines)}")
        # Pad or truncate
        while len(lines) < 3:
            lines.append(lines[0] if lines else "")
        lines = lines[:3]
    
    return lines


def main():
    """Generate V6 L3 long descriptions."""
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    
    print("="*70)
    print("V6_L3: LONG DESCRIPTIONS (3 variations per label)")
    print("="*70)
    
    v6_l3_results = {}
    
    for label_id, label_name in AG_NEWS_LABELS.items():
        print(f"\n{label_id}: {label_name}")
        lines = generate_v6_l3(label_name, client, model)
        
        v6_l3_results[label_id] = lines
        
        for i, line in enumerate(lines, 1):
            print(f"   {i}. {line}")
    
    # Save in both formats
    output_dir = Path("scripts/prompt_engineering/results")
    
    # Raw format
    output_file = output_dir / "ag_news_descriptions_v6_l3_long.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(v6_l3_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Raw saved: {output_file}")
    
    # Combined L2+L3 format for testing
    # Need to load L2 from previous V6 results
    v6_l2_file = output_dir / "ag_news_descriptions_v6_simple.json"
    if v6_l2_file.exists():
        with open(v6_l2_file) as f:
            v6_l2 = json.load(f)
        
        combined = {
            "ag_news": {
                str(k): {
                    "l2": v6_l2[str(k)] if str(k) in v6_l2 else v6_l2[k],
                    "l3": v6_l3_results[k]
                }
                for k in v6_l3_results.keys()
            }
        }
        
        combined_file = output_dir / "ag_news_descriptions_v6_l2_l3_combined.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"✅ Combined L2+L3: {combined_file}")
    
    print("\n" + "="*70)
    print("✅ V6_L3 LONG DESCRIPTIONS GENERATED!")
    print("="*70)
    print("\nNext:")
    print("1. Copy to src/label_descriptions/")
    print("2. Test with l3 mode")
    print("3. Compare: L1 vs L2 vs L3")


if __name__ == "__main__":
    main()