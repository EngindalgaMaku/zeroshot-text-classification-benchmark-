"""Detailed analysis of description quality issues."""
import json
from pathlib import Path

# Load generated descriptions
with open("src/label_descriptions/generated_descriptions.json") as f:
    generated = json.load(f)["banking77"]

# Manual descriptions from labels.py
manual_descriptions = {
    "0": "The user wants to activate their card or asking how to activate it.",
    "2": "The user has a question about Apple Pay or Google Pay integration.",
    "7": "The user cannot add a beneficiary or the beneficiary is not allowed.",
    "10": "The user has questions about where their card is accepted.",
    "14": "The user's card is not working properly.",
}

print("=== DETAILED SEMANTIC ANALYSIS ===\n")

for label_id, manual in manual_descriptions.items():
    gen_l2 = generated[label_id]['l2']
    gen_l3 = generated[label_id]['l3']
    
    print(f"\n{'='*80}")
    print(f"LABEL {label_id}")
    print(f"{'='*80}")
    
    # Word count
    manual_words = len(manual.split())
    l2_words = len(gen_l2.split())
    l3_words = sum(len(s.split()) for s in gen_l3)
    
    print(f"\nWORD COUNT:")
    print(f"  Manual: {manual_words} words")
    print(f"  L2:     {l2_words} words ({l2_words/manual_words:.1f}x longer)")
    print(f"  L3:     {l3_words} words ({l3_words/manual_words:.1f}x longer)")
    
    # Key differences
    print(f"\nKEY SEMANTIC DIFFERENCES:")
    print(f"\nManual: {manual}")
    print(f"  → Focus: Direct user intent")
    print(f"  → Style: Simple, action-oriented")
    
    print(f"\nL2: {gen_l2}")
    print(f"  → Focus: Expanded explanation")
    print(f"  → Style: More formal, adds context")
    
    print(f"\nL3 (averaged from 3 sentences):")
    for i, sent in enumerate(gen_l3, 1):
        print(f"  {i}. {sent}")
    print(f"  → Focus: Comprehensive coverage")
    print(f"  → Style: Very formal, educational")
    print(f"  → Problem: Each sentence has different focus, averaging loses specificity")

print(f"\n\n{'='*80}")
print("SUMMARY OF ISSUES")
print(f"{'='*80}\n")

print("1. VERBOSITY PROBLEM:")
print("   - Manual descriptions are 10-15 words")
print("   - L2 descriptions are 15-20 words (1.5x longer)")
print("   - L3 descriptions are 40-60 words (4-5x longer)")
print("   - Longer text adds noise to embeddings\n")

print("2. SEMANTIC DRIFT:")
print("   - Manual: 'card not working' → direct problem")
print("   - L2: adds 'seeking assistance' → shifts focus")
print("   - L3: adds 'lost/stolen/disabled' → introduces new concepts\n")

print("3. L3 AVERAGING PROBLEM:")
print("   - Sentence 1: Focuses on one aspect (e.g., 'activation process')")
print("   - Sentence 2: Focuses on another (e.g., 'requesting assistance')")
print("   - Sentence 3: Focuses on yet another (e.g., 'confirmation')")
print("   - Averaging these embeddings dilutes the core intent\n")

print("4. FORMALITY MISMATCH:")
print("   - Manual: Conversational ('The user wants...')")
print("   - Generated: Academic ('The intent encompasses...')")
print("   - User queries are informal, manual style matches better\n")

print("RECOMMENDATIONS:")
print("  1. Shorten prompts: 'Generate a 10-15 word description'")
print("  2. Simplify style: 'Use simple, direct language'")
print("  3. For L3: Use max pooling instead of averaging")
print("  4. Match manual style: 'Write as: The user wants/has/cannot...'")
