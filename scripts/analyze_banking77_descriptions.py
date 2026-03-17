"""Analyze banking77 description quality and performance differences."""
import json
from pathlib import Path

# Load generated descriptions
with open("src/label_descriptions/generated_descriptions.json") as f:
    generated = json.load(f)["banking77"]

# Sample labels to analyze
sample_labels = ["0", "2", "7", "10", "14"]  # Different performance levels

print("=== Banking77 Description Comparison ===\n")
print("Results: Manual (63.1%) > L2 (60.5%) > L3 (57.2%)\n")

# Manual descriptions from labels.py
manual_descriptions = {
    "0": "The user wants to activate their card or asking how to activate it.",
    "2": "The user has a question about Apple Pay or Google Pay integration.",
    "7": "The user cannot add a beneficiary or the beneficiary is not allowed.",
    "10": "The user has questions about where their card is accepted.",
    "14": "The user's card is not working properly.",
}

print("=" * 80)
for label_id in sample_labels:
    print(f"\nLabel {label_id}: {list(generated[label_id].keys())}")
    print(f"\nManual:")
    print(f"  {manual_descriptions[label_id]}")
    print(f"\nGenerated L2:")
    print(f"  {generated[label_id]['l2']}")
    print(f"\nGenerated L3 (3 sentences):")
    for i, sent in enumerate(generated[label_id]['l3'], 1):
        print(f"  {i}. {sent}")
    print("\n" + "=" * 80)

print("\n=== KEY FINDINGS ===\n")
print("1. LENGTH DIFFERENCE:")
print("   - Manual: Short, concise (10-15 words)")
print("   - L2: Longer, more detailed (15-20 words)")
print("   - L3: Much longer (3 sentences, 40-60 words total)")
print()
print("2. STYLE DIFFERENCE:")
print("   - Manual: Direct, simple language")
print("   - Generated: More formal, verbose, explanatory")
print()
print("3. WHY L3 < L2:")
print("   - L3 has 3 separate embeddings that get averaged")
print("   - Averaging dilutes the semantic signal")
print("   - More text = more noise in embedding space")
print()
print("4. WHY BOTH < MANUAL:")
print("   - Manual descriptions are optimized for embedding models")
print("   - Generated descriptions are more natural but less optimized")
print("   - LLM adds explanatory context that may confuse embeddings")
print()
print("5. POTENTIAL FIXES:")
print("   - Adjust prompts to generate shorter, more concise descriptions")
print("   - Use max pooling instead of averaging for L3")
print("   - Fine-tune prompt to match manual style better")
