"""Evaluate candidate datasets for inclusion in the benchmark.

This script evaluates SST-2, TREC, and IMDB datasets across multiple criteria:
- Task diversity contribution
- Dataset quality and reliability
- Public availability and ease of use
- Class balance
- Text characteristics

Requirements: 11.2, 11.5
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_candidate_datasets():
    """Evaluate candidate datasets against inclusion criteria."""
    
    print("=" * 80)
    print("CANDIDATE DATASET EVALUATION")
    print("=" * 80)
    print()
    
    # Define candidate datasets with detailed characteristics
    candidates = {
        "sst2": {
            "full_name": "Stanford Sentiment Treebank (SST-2)",
            "hf_dataset": "glue/sst2",
            "task_type": "sentiment_classification",
            "num_classes": 2,
            "class_names": ["negative", "positive"],
            "domain": "movie_reviews",
            "text_length": "short",
            "avg_tokens": 19,
            "test_size": 1821,
            "train_size": 67349,
            "class_balance": "balanced",
            "quality": "high",
            "community_adoption": "very_high",
            "benchmark_status": "standard",
            "pros": [
                "Well-established benchmark (GLUE task)",
                "Binary sentiment - fills gap in current benchmark",
                "High quality annotations",
                "Balanced classes",
                "Movie domain complements financial sentiment"
            ],
            "cons": [
                "Short texts (single sentences)",
                "Limited to movie reviews domain",
                "Binary only (less challenging than multi-class)"
            ],
            "diversity_contribution": {
                "task_type": "HIGH - adds general sentiment (vs. financial only)",
                "domain": "HIGH - movie reviews vs. financial news",
                "class_count": "HIGH - adds binary classification",
                "text_length": "LOW - similar to existing short datasets"
            }
        },
        "trec": {
            "full_name": "TREC Question Classification",
            "hf_dataset": "trec",
            "task_type": "question_classification",
            "num_classes": 6,
            "class_names": ["abbreviation", "entity", "description", "human", "location", "numeric"],
            "domain": "questions",
            "text_length": "short",
            "avg_tokens": 10,
            "test_size": 500,
            "train_size": 5452,
            "class_balance": "imbalanced",
            "quality": "high",
            "community_adoption": "high",
            "benchmark_status": "standard",
            "pros": [
                "Classic question classification benchmark",
                "Focuses on question types (vs. topics in Yahoo)",
                "Well-studied dataset",
                "Small test set (500) - fast evaluation"
            ],
            "cons": [
                "Overlap with Yahoo Answers (both Q&A domain)",
                "Very short texts (questions only)",
                "Small test set may limit statistical power",
                "Imbalanced classes"
            ],
            "diversity_contribution": {
                "task_type": "LOW - Yahoo Answers already covers Q&A",
                "domain": "LOW - questions domain already represented",
                "class_count": "LOW - 6 classes similar to existing",
                "text_length": "LOW - very short, similar to existing"
            }
        },
        "imdb": {
            "full_name": "IMDB Movie Reviews",
            "hf_dataset": "imdb",
            "task_type": "sentiment_classification",
            "num_classes": 2,
            "class_names": ["negative", "positive"],
            "domain": "movie_reviews",
            "text_length": "long",
            "avg_tokens": 233,
            "test_size": 25000,
            "train_size": 25000,
            "class_balance": "balanced",
            "quality": "high",
            "community_adoption": "very_high",
            "benchmark_status": "standard",
            "pros": [
                "Long-form text classification (avg 233 tokens)",
                "Binary sentiment - fills gap",
                "Large test set (25k) - strong statistical power",
                "Perfectly balanced classes",
                "Complements SST-2 (same domain, different length)"
            ],
            "cons": [
                "Same domain as SST-2 (movie reviews)",
                "Binary only (less challenging)",
                "Large test set may slow evaluation (can sample)"
            ],
            "diversity_contribution": {
                "task_type": "HIGH - adds general sentiment",
                "domain": "HIGH - movie reviews (new domain)",
                "class_count": "HIGH - adds binary classification",
                "text_length": "VERY HIGH - long texts (233 tokens vs. 19 for SST-2)"
            }
        }
    }
    
    # Evaluate each candidate
    for ds_key, info in candidates.items():
        print("=" * 80)
        print(f"CANDIDATE: {info['full_name']}")
        print("=" * 80)
        print()
        
        print("BASIC INFORMATION")
        print("-" * 80)
        print(f"HuggingFace Dataset: {info['hf_dataset']}")
        print(f"Task Type: {info['task_type'].replace('_', ' ').title()}")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"Class Names: {', '.join(info['class_names'])}")
        print(f"Domain: {info['domain'].replace('_', ' ').title()}")
        print(f"Text Length: {info['text_length'].title()} (avg {info['avg_tokens']} tokens)")
        print()
        
        print("DATASET STATISTICS")
        print("-" * 80)
        print(f"Test Set Size: {info['test_size']:,}")
        print(f"Train Set Size: {info['train_size']:,}")
        print(f"Class Balance: {info['class_balance'].title()}")
        print(f"Quality: {info['quality'].replace('_', ' ').title()}")
        print(f"Community Adoption: {info['community_adoption'].replace('_', ' ').title()}")
        print(f"Benchmark Status: {info['benchmark_status'].title()}")
        print()
        
        print("DIVERSITY CONTRIBUTION")
        print("-" * 80)
        for criterion, contribution in info['diversity_contribution'].items():
            print(f"{criterion.replace('_', ' ').title()}: {contribution}")
        print()
        
        print("PROS")
        print("-" * 80)
        for i, pro in enumerate(info['pros'], 1):
            print(f"{i}. {pro}")
        print()
        
        print("CONS")
        print("-" * 80)
        for i, con in enumerate(info['cons'], 1):
            print(f"{i}. {con}")
        print()
        print()
    
    # Comparative analysis
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    print("1. TASK DIVERSITY CONTRIBUTION")
    print("-" * 80)
    print("SST-2:  HIGH - Adds binary sentiment (movie domain)")
    print("TREC:   LOW  - Question classification overlaps with Yahoo Answers")
    print("IMDB:   HIGH - Adds binary sentiment + long text classification")
    print()
    
    print("2. TEXT LENGTH DIVERSITY")
    print("-" * 80)
    print("SST-2:  LOW  - Short texts (19 tokens), similar to existing")
    print("TREC:   LOW  - Very short texts (10 tokens), similar to existing")
    print("IMDB:   VERY HIGH - Long texts (233 tokens), only 20 Newsgroups is long")
    print()
    
    print("3. DOMAIN DIVERSITY")
    print("-" * 80)
    print("SST-2:  HIGH - Movie reviews (new domain)")
    print("TREC:   LOW  - Questions (Yahoo Answers already covers Q&A)")
    print("IMDB:   HIGH - Movie reviews (new domain)")
    print()
    
    print("4. CLASS COUNT DIVERSITY")
    print("-" * 80)
    print("SST-2:  HIGH - Binary (only Twitter Financial has ≤3 classes)")
    print("TREC:   LOW  - 6 classes (similar to AG News with 4)")
    print("IMDB:   HIGH - Binary (fills gap)")
    print()
    
    print("5. QUALITY & ADOPTION")
    print("-" * 80)
    print("SST-2:  VERY HIGH - GLUE benchmark, widely used")
    print("TREC:   HIGH - Classic benchmark, well-studied")
    print("IMDB:   VERY HIGH - Standard sentiment benchmark")
    print()
    
    # Scoring
    print("=" * 80)
    print("INCLUSION SCORING")
    print("=" * 80)
    print()
    
    scores = {
        "sst2": {
            "task_diversity": 9,
            "domain_diversity": 9,
            "class_diversity": 9,
            "text_length_diversity": 3,
            "quality": 10,
            "adoption": 10,
            "total": 0
        },
        "trec": {
            "task_diversity": 3,
            "domain_diversity": 3,
            "class_diversity": 4,
            "text_length_diversity": 2,
            "quality": 9,
            "adoption": 8,
            "total": 0
        },
        "imdb": {
            "task_diversity": 9,
            "domain_diversity": 9,
            "class_diversity": 9,
            "text_length_diversity": 10,
            "quality": 10,
            "adoption": 10,
            "total": 0
        }
    }
    
    # Calculate totals
    for ds in scores:
        scores[ds]["total"] = sum(v for k, v in scores[ds].items() if k != "total")
    
    print("Scoring Criteria (out of 10):")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Task':<6} {'Domain':<8} {'Classes':<9} {'Length':<8} {'Quality':<9} {'Adoption':<10} {'TOTAL':<6}")
    print("-" * 80)
    
    for ds_key in ["sst2", "trec", "imdb"]:
        s = scores[ds_key]
        print(f"{ds_key.upper():<15} {s['task_diversity']:<6} {s['domain_diversity']:<8} {s['class_diversity']:<9} {s['text_length_diversity']:<8} {s['quality']:<9} {s['adoption']:<10} {s['total']:<6}")
    
    print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    print("RANK 1: IMDB (Score: 57/60)")
    print("-" * 80)
    print("✅ STRONGLY RECOMMENDED")
    print()
    print("Rationale:")
    print("  - Fills TWO critical gaps: sentiment diversity + long text classification")
    print("  - Only dataset with long texts besides 20 Newsgroups")
    print("  - Adds binary classification (only Twitter Financial has ≤3 classes)")
    print("  - High quality, widely adopted, standard benchmark")
    print("  - Large test set (25k) provides strong statistical power")
    print()
    print("Impact on Benchmark:")
    print("  - Sentiment datasets: 1 → 2 (financial + movie)")
    print("  - Binary classification: 1 → 2 (Twitter Financial + IMDB)")
    print("  - Long text datasets: 1 → 2 (20 Newsgroups + IMDB)")
    print("  - Total datasets: 7 → 8")
    print()
    
    print("RANK 2: SST-2 (Score: 50/60)")
    print("-" * 80)
    print("✅ RECOMMENDED (if adding 2 datasets)")
    print()
    print("Rationale:")
    print("  - Fills sentiment diversity gap (movie vs. financial)")
    print("  - Adds binary classification")
    print("  - GLUE benchmark - very high adoption")
    print("  - Complements IMDB (same domain, different length)")
    print("  - Fast evaluation (1.8k test samples)")
    print()
    print("Impact on Benchmark:")
    print("  - Sentiment datasets: 2 → 3 (financial + movie short + movie long)")
    print("  - Binary classification: 2 → 3")
    print("  - Total datasets: 8 → 9")
    print()
    print("Note: Only add if IMDB is also added (they complement each other)")
    print()
    
    print("RANK 3: TREC (Score: 29/60)")
    print("-" * 80)
    print("❌ NOT RECOMMENDED")
    print()
    print("Rationale:")
    print("  - Low diversity contribution (overlaps with Yahoo Answers)")
    print("  - Very short texts (10 tokens) - no length diversity")
    print("  - Small test set (500) may limit statistical power")
    print("  - 6 classes similar to existing datasets")
    print()
    print("Verdict: Skip TREC - insufficient diversity contribution")
    print()
    
    # Final recommendation
    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print()
    print("OPTION 1: Add IMDB only (RECOMMENDED)")
    print("-" * 80)
    print("  - Addresses the two most critical gaps: sentiment + long text")
    print("  - Minimal expansion (7 → 8 datasets)")
    print("  - High impact per dataset added")
    print("  - Keeps benchmark focused and manageable")
    print()
    
    print("OPTION 2: Add IMDB + SST-2 (ALTERNATIVE)")
    print("-" * 80)
    print("  - Comprehensive sentiment coverage (financial + movie short + movie long)")
    print("  - Strong binary classification representation (3 datasets)")
    print("  - SST-2 adds GLUE benchmark prestige")
    print("  - Moderate expansion (7 → 9 datasets)")
    print("  - More experiments to run (9 datasets × 7 models = 63 total)")
    print()
    
    print("OPTION 3: No expansion (ACCEPTABLE)")
    print("-" * 80)
    print("  - Current 7 datasets provide adequate coverage")
    print("  - 5 task types represented")
    print("  - Sentiment gap is not critical (1 dataset exists)")
    print("  - Focus on analysis depth rather than breadth")
    print()
    
    print("=" * 80)
    print("DECISION GUIDANCE")
    print("=" * 80)
    print("""
For TACL submission, consider:

1. If reviewers might question sentiment coverage → Add IMDB
2. If you want comprehensive sentiment analysis → Add IMDB + SST-2
3. If current scope is sufficient → No expansion

RECOMMENDED: Add IMDB only
  - Strongest single addition (fills 2 gaps)
  - Manageable scope increase (8 datasets vs. 7)
  - Clear justification for inclusion
  - Can always add SST-2 later if reviewers request it
""")
    
    print("=" * 80)
    print("END OF EVALUATION")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_candidate_datasets()
