"""Generate V6 L2+L3 descriptions for ALL datasets."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Dataset configurations
DATASETS = {
    "ag_news": {
        "domain": "news articles",
        "labels": {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
    },
    "dbpedia_14": {
        "domain": "encyclopedia entities",
        "labels": {
            0: "Company", 1: "EducationalInstitution", 2: "Artist", 3: "Athlete",
            4: "OfficeHolder", 5: "MeanOfTransportation", 6: "Building",
            7: "NaturalPlace", 8: "Village", 9: "Animal", 10: "Plant",
            11: "Album", 12: "Film", 13: "WrittenWork"
        }
    },
    "yahoo_answers_topics": {
        "domain": "community questions",
        "labels": {
            0: "society and culture", 1: "science and mathematics", 2: "health",
            3: "education and reference", 4: "computers and internet", 5: "sports",
            6: "business and finance", 7: "entertainment and music",
            8: "family and relationships", 9: "politics and government"
        }
    },
    "banking77": {
        "domain": "banking customer service",
        "labels": {i: label for i, label in enumerate([
            "activate my card", "age limit", "apple pay or google pay", "atm support",
            "automatic top up", "balance not updated after bank transfer",
            "balance not updated after cheque or cash deposit", "beneficiary not allowed",
            "cancel transfer", "card about to expire", "card acceptance", "card arrival",
            "card delivery estimate", "card linking", "card not working",
            "card payment fee charged", "card payment not recognised",
            "card payment wrong exchange rate", "card swallowed", "cash withdrawal charge",
            "cash withdrawal not recognised", "change pin", "compromised card",
            "contactless not working", "country support", "declined card payment",
            "declined cash withdrawal", "declined transfer",
            "direct debit payment not recognised", "disposable card limits",
            "edit personal details", "exchange charge", "exchange rate", "exchange via app",
            "extra charge on statement", "failed transfer", "fiat currency support",
            "get disposable virtual card", "get physical card", "getting spare card",
            "getting virtual card", "lost or stolen card", "lost or stolen phone",
            "order physical card", "passcode forgotten", "pending card payment",
            "pending cash withdrawal", "pending top up", "pending transfer", "pin blocked",
            "receiving money", "refund not showing up", "request refund",
            "reverted card payment?", "supported cards and currencies", "terminate account",
            "top up by bank transfer charge", "top up by card charge",
            "top up by cash or cheque", "top up failed", "top up limits", "top up reverted",
            "topping up by card", "transaction charged twice", "transfer fee charged",
            "transfer into account", "transfer not received by recipient", "transfer timing",
            "unable to verify identity", "verify my identity", "verify source of funds",
            "verify top up", "virtual card not working", "visa or mastercard",
            "why verify identity", "wrong amount of cash received",
            "wrong exchange rate for cash withdrawal"
        ])}
    },
    "zeroshot/twitter-financial-news-sentiment": {
        "domain": "financial market sentiment",
        "labels": {
            0: "bearish",
            1: "bullish",
            2: "neutral"
        }
    },
    "SetFit/20_newsgroups": {
        "domain": "online discussion forums",
        "labels": {
            0: "alt.atheism", 1: "comp.graphics", 2: "comp.os.ms-windows.misc",
            3: "comp.sys.ibm.pc.hardware", 4: "comp.sys.mac.hardware", 5: "comp.windows.x",
            6: "misc.forsale", 7: "rec.autos", 8: "rec.motorcycles",
            9: "rec.sport.baseball", 10: "rec.sport.hockey", 11: "sci.crypt",
            12: "sci.electronics", 13: "sci.med", 14: "sci.space",
            15: "soc.religion.christian", 16: "talk.politics.guns",
            17: "talk.politics.mideast", 18: "talk.politics.misc", 19: "talk.religion.misc"
        }
    },
    "imdb": {
        "domain": "movie reviews",
        "labels": {
            0: "neg",
            1: "pos"
        }
    },
    "sst2": {
        "domain": "sentiment analysis",
        "labels": {
            0: "negative",
            1: "positive"
        }
    },
    "go_emotions": {
        "domain": "emotional expressions",
        "labels": {
            0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
            5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
            10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
            14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
            19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
            23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
        }
    }
}

# V6 L2 Prompt
PROMPT_V6_L2 = """You are generating a text classification description for {dataset_domain}.

Dataset: {dataset_name}
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
- Use {dataset_domain} domain terminology
- Use concrete nouns (NOT abstract concepts)
- Each term must be specific to {label_name} in {dataset_domain}
- NO generic words (events, issues, topics, things, aspects, various)

Now generate ONLY the single sentence for:
Label: {label_name}
Domain: {dataset_domain}

Output:"""

# V6 L3 Prompt
PROMPT_V6_L3 = """You are generating a comprehensive text classification description for {dataset_domain}.

Dataset: {dataset_name}
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

Now generate 3 lines for:
Label: {label_name}
Domain: {dataset_domain}

Output (3 lines):"""


def generate_l2(dataset_name, domain, label_name, client, model):
    """Generate L2 description."""
    prompt = PROMPT_V6_L2.format(
        dataset_name=dataset_name,
        dataset_domain=domain,
        label_name=label_name
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    
    return response.choices[0].message.content.strip()


def generate_l3(dataset_name, domain, label_name, client, model):
    """Generate L3 description (3 variations)."""
    prompt = PROMPT_V6_L3.format(
        dataset_name=dataset_name,
        dataset_domain=domain,
        label_name=label_name
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    
    text = response.choices[0].message.content.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Ensure exactly 3 lines
    while len(lines) < 3:
        lines.append(lines[0] if lines else "")
    return lines[:3]


def main():
    """Generate V6 L2+L3 for all datasets."""
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    
    all_descriptions = {}
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name} ({config['domain']})")
        print('='*70)
        
        dataset_desc = {}
        
        for label_id, label_name in config["labels"].items():
            print(f"\n{label_id}: {label_name}")
            
            # Generate L2
            l2 = generate_l2(dataset_name, config["domain"], label_name, client, model)
            print(f"   L2: {l2}")
            
            # Generate L3
            l3 = generate_l3(dataset_name, config["domain"], label_name, client, model)
            print(f"   L3:")
            for i, line in enumerate(l3, 1):
                print(f"      {i}. {line}")
            
            dataset_desc[str(label_id)] = {
                "l2": l2,
                "l3": l3
            }
        
        all_descriptions[dataset_name] = dataset_desc
    
    # Save
    output_file = Path("src/label_descriptions/generated_descriptions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ ALL V6 L2+L3 DESCRIPTIONS GENERATED!")
    print("="*70)
    print(f"\nSaved to: {output_file}")
    print(f"\nTotal datasets: {len(all_descriptions)}")
    print(f"Total labels: {sum(len(d) for d in all_descriptions.values())}")
    
    print("\n✅ Ready for experiments!")
    print("Use label_mode='l2' or 'l3' in config files")


if __name__ == "__main__":
    main()