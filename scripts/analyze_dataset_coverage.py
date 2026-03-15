"""Analyze current dataset coverage and identify gaps in task-type diversity.

This script evaluates the 7 current benchmark datasets across multiple dimensions:
- Task type distribution (sentiment, topic, entity, intent, emotion)
- Number of classes per dataset
- Text length characteristics
- Domain diversity

Requirements: 11.3
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels import LABEL_SETS


def analyze_dataset_coverage():
    """Analyze current dataset coverage across task types."""
    
    print("=" * 80)
    print("DATASET COVERAGE ANALYSIS")
    print("=" * 80)
    print()
    
    # Define current datasets with their characteristics
    datasets = {
        "ag_news": {
            "task_type": "topic_classification",
            "num_classes": 4,
            "domain": "news",
            "text_length": "short",
            "description": "News article topic classification (world, sports, business, sci/tech)"
        },
        "dbpedia_14": {
            "task_type": "entity_classification",
            "num_classes": 14,
            "domain": "knowledge_base",
            "text_length": "medium",
            "description": "Entity type classification (company, person, place, etc.)"
        },
        "yahoo_answers_topics": {
            "task_type": "topic_classification",
            "num_classes": 10,
            "domain": "qa_forum",
            "text_length": "medium",
            "description": "Question topic classification (society, science, health, etc.)"
        },
        "banking77": {
            "task_type": "intent_classification",
            "num_classes": 77,
            "domain": "banking",
            "text_length": "short",
            "description": "Banking customer intent classification (fine-grained)"
        },
        "zeroshot/twitter-financial-news-sentiment": {
            "task_type": "sentiment_classification",
            "num_classes": 3,
            "domain": "financial_news",
            "text_length": "short",
            "description": "Financial sentiment (bearish, bullish, neutral)"
        },
        "SetFit/20_newsgroups": {
            "task_type": "topic_classification",
            "num_classes": 20,
            "domain": "discussion_forums",
            "text_length": "long",
            "description": "Newsgroup topic classification (diverse technical and social topics)"
        },
        "go_emotions": {
            "task_type": "emotion_classification",
            "num_classes": 28,
            "domain": "social_media",
            "text_length": "short",
            "description": "Fine-grained emotion classification (27 emotions + neutral)"
        }
    }
    
    # 1. Task Type Distribution
    print("1. TASK TYPE DISTRIBUTION")
    print("-" * 80)
    task_types = {}
    for ds_name, info in datasets.items():
        task_type = info["task_type"]
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(ds_name)
    
    for task_type, ds_list in sorted(task_types.items()):
        print(f"\n{task_type.replace('_', ' ').title()}: {len(ds_list)} dataset(s)")
        for ds in ds_list:
            print(f"  - {ds}: {datasets[ds]['description']}")
    
    print("\n" + "=" * 80)
    print("TASK TYPE SUMMARY")
    print("=" * 80)
    for task_type, ds_list in sorted(task_types.items()):
        print(f"{task_type.replace('_', ' ').title()}: {len(ds_list)}")
    
    # 2. Class Count Distribution
    print("\n\n2. CLASS COUNT DISTRIBUTION")
    print("-" * 80)
    class_counts = [(ds, info["num_classes"]) for ds, info in datasets.items()]
    class_counts.sort(key=lambda x: x[1])
    
    for ds, count in class_counts:
        print(f"{ds:50s} {count:3d} classes")
    
    avg_classes = sum(c for _, c in class_counts) / len(class_counts)
    print(f"\nAverage: {avg_classes:.1f} classes")
    print(f"Range: {class_counts[0][1]} - {class_counts[-1][1]} classes")
    
    # 3. Domain Diversity
    print("\n\n3. DOMAIN DIVERSITY")
    print("-" * 80)
    domains = {}
    for ds_name, info in datasets.items():
        domain = info["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(ds_name)
    
    for domain, ds_list in sorted(domains.items()):
        print(f"\n{domain.replace('_', ' ').title()}: {len(ds_list)} dataset(s)")
        for ds in ds_list:
            print(f"  - {ds}")
    
    # 4. Text Length Distribution
    print("\n\n4. TEXT LENGTH DISTRIBUTION")
    print("-" * 80)
    text_lengths = {}
    for ds_name, info in datasets.items():
        length = info["text_length"]
        if length not in text_lengths:
            text_lengths[length] = []
        text_lengths[length].append(ds_name)
    
    for length in ["short", "medium", "long"]:
        if length in text_lengths:
            print(f"\n{length.title()}: {len(text_lengths[length])} dataset(s)")
            for ds in text_lengths[length]:
                print(f"  - {ds}")
    
    # 5. Gap Analysis
    print("\n\n5. GAP ANALYSIS")
    print("=" * 80)
    
    gaps = []
    
    # Check task type coverage
    if len(task_types.get("sentiment_classification", [])) == 1:
        gaps.append("⚠️  SENTIMENT: Only 1 dataset (Twitter Financial). Limited sentiment diversity.")
    
    if len(task_types.get("emotion_classification", [])) == 1:
        gaps.append("✅ EMOTION: 1 dataset (GoEmotions). Adequate for fine-grained emotion.")
    
    if len(task_types.get("intent_classification", [])) == 1:
        gaps.append("✅ INTENT: 1 dataset (Banking77). Adequate for intent classification.")
    
    if len(task_types.get("entity_classification", [])) == 1:
        gaps.append("✅ ENTITY: 1 dataset (DBPedia). Adequate for entity classification.")
    
    if len(task_types.get("topic_classification", [])) >= 3:
        gaps.append("✅ TOPIC: 3 datasets (AG News, Yahoo, 20 Newsgroups). Good coverage.")
    
    # Check class count diversity
    binary_count = sum(1 for _, c in class_counts if c <= 3)
    if binary_count == 1:
        gaps.append("⚠️  BINARY/TERNARY: Only 1 dataset with ≤3 classes. Limited for binary tasks.")
    
    # Check text length diversity
    if len(text_lengths.get("long", [])) == 1:
        gaps.append("⚠️  LONG TEXT: Only 1 dataset with long texts. Limited for document classification.")
    
    print("\nCoverage Assessment:")
    for gap in gaps:
        print(f"  {gap}")
    
    # 6. Recommendations
    print("\n\n6. RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Sentiment gap
    if len(task_types.get("sentiment_classification", [])) == 1:
        recommendations.append({
            "priority": "HIGH",
            "gap": "Sentiment diversity",
            "suggestion": "Add SST-2 (binary sentiment) or IMDB (binary, longer texts)",
            "rationale": "Current sentiment coverage limited to financial domain only"
        })
    
    # Binary classification gap
    if binary_count == 1:
        recommendations.append({
            "priority": "MEDIUM",
            "gap": "Binary classification",
            "suggestion": "Add SST-2 or IMDB (both binary sentiment)",
            "rationale": "Binary classification is common in practice, needs better representation"
        })
    
    # Long text gap
    if len(text_lengths.get("long", [])) == 1:
        recommendations.append({
            "priority": "MEDIUM",
            "gap": "Long text classification",
            "suggestion": "Add IMDB (movie reviews, longer texts)",
            "rationale": "Only 20 Newsgroups has long texts, IMDB would add sentiment + length diversity"
        })
    
    # Question classification
    recommendations.append({
        "priority": "LOW",
        "gap": "Question classification diversity",
        "suggestion": "Add TREC (6-class question classification)",
        "rationale": "Yahoo Answers covers Q&A topics, but TREC focuses on question types"
    })
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['gap']}")
        print(f"   Suggestion: {rec['suggestion']}")
        print(f"   Rationale: {rec['rationale']}")
    
    # 7. Summary
    print("\n\n7. SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {len(datasets)}")
    print(f"Task types covered: {len(task_types)}")
    print(f"  - Topic classification: {len(task_types.get('topic_classification', []))} (STRONG)")
    print(f"  - Sentiment classification: {len(task_types.get('sentiment_classification', []))} (WEAK)")
    print(f"  - Entity classification: {len(task_types.get('entity_classification', []))} (ADEQUATE)")
    print(f"  - Intent classification: {len(task_types.get('intent_classification', []))} (ADEQUATE)")
    print(f"  - Emotion classification: {len(task_types.get('emotion_classification', []))} (ADEQUATE)")
    print(f"\nClass count range: {class_counts[0][1]} - {class_counts[-1][1]}")
    print(f"Domain diversity: {len(domains)} unique domains")
    print(f"Text length diversity: {len(text_lengths)} categories")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The current 7-dataset benchmark provides:
✅ STRONG coverage of topic classification (3 datasets)
✅ ADEQUATE coverage of entity, intent, and emotion classification (1 each)
⚠️  WEAK coverage of sentiment classification (1 dataset, financial domain only)
⚠️  LIMITED binary classification representation (1 dataset)
⚠️  LIMITED long text representation (1 dataset)

RECOMMENDATION: Consider adding 1-2 datasets to address sentiment diversity and 
binary classification gaps. Top candidates:
  1. SST-2: Binary sentiment, movie reviews, well-established benchmark
  2. IMDB: Binary sentiment, longer texts, complements SST-2
  3. TREC: Question classification, adds question-type diversity (lower priority)
""")
    
    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    analyze_dataset_coverage()
