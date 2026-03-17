# Optimized Prompts Testing Report - Banking77

## Executive Summary

Tested optimized prompts that generate shorter, more concise label descriptions. Results show that **shorter descriptions led to decreased performance**, suggesting that the additional context in longer descriptions is beneficial for zero-shot classification.

## Prompt Changes

### Old Prompts (15-20 words target)
- **L2**: "Define the following text classification label in 15–20 words..."
- **L3**: "Generate three different short descriptions... Keep each description between 15–20 words..."

### New Prompts (10-15 words target)
- **L2**: "Define the following text classification label in 10–15 words..."
- **L3**: "Generate three different short descriptions... Keep each description between 10–15 words..."

## Word Count Analysis

### Sample Labels Comparison

| Label | Old L2 (words) | New L2 (words) | Change |
|-------|----------------|----------------|--------|
| 0 (activate_my_card) | 15 | 8 | -7 |
| 2 (apple_pay_or_google_pay) | 15 | 13 | -2 |
| 7 (beneficiary_not_allowed) | 18 | 11 | -7 |
| 10 (card_acceptance) | 18 | 12 | -6 |
| 14 (card_not_working) | 18 | 12 | -6 |

### Overall Statistics

- **Old Average**: 16.7 words per L2 description
- **New Average**: 10.3 words per L2 description
- **Reduction**: -6.4 words (38% decrease)

## Performance Results

### L2 Descriptions (Single Description per Label)

| Metric | Old (15-20 words) | New (10-15 words) | Change |
|--------|-------------------|-------------------|--------|
| **Accuracy** | 60.5% | 57.5% | **-3.0%** ❌ |
| Macro F1 | 59.3% | 55.9% | -3.4% |
| Weighted F1 | 59.8% | 56.7% | -3.1% |
| Macro Precision | 66.4% | 63.4% | -3.0% |
| Macro Recall | 59.3% | 57.1% | -2.2% |

### L3 Descriptions (Three Descriptions per Label)

| Metric | Old (15-20 words) | New (10-15 words) | Change |
|--------|-------------------|---|--------|
| **Accuracy** | 57.2% | 56.3% | **-0.9%** ❌ |
| Macro F1 | 54.5% | 53.5% | -1.0% |
| Weighted F1 | 55.3% | 54.4% | -0.9% |
| Macro Precision | 61.0% | 60.2% | -0.8% |
| Macro Recall | 56.8% | 56.1% | -0.7% |

## Example Description Comparisons

### Label 0: activate_my_card

**Old (15 words):**
> The user wants to activate their card to enable transactions and access their account services.

**New (8 words):**
> The user wants to activate their bank card.

**Analysis:** The new description is more concise but loses context about *why* activation is needed (transactions, account access).

### Label 7: beneficiary_not_allowed

**Old (18 words):**
> The user is inquiring about a situation where a beneficiary is not allowed to receive funds or benefits.

**New (11 words):**
> The user wants to know why a beneficiary is not allowed.

**Analysis:** The new description is shorter but loses the important context about "funds or benefits."

### Label 10: card_acceptance

**Old (18 words):**
> The user wants to confirm that their card is accepted for transactions at various merchants or service providers.

**New (12 words):**
> The user wants to know if their card is accepted for transactions.

**Analysis:** The new description removes "various merchants or service providers" which provides useful context.

## Key Findings

### 1. Shorter ≠ Better
- **38% reduction in word count** led to **3.0% drop in L2 accuracy**
- The additional context in longer descriptions appears to help the model better distinguish between similar labels

### 2. L2 More Affected Than L3
- L2 accuracy dropped by 3.0 percentage points
- L3 accuracy dropped by only 0.9 percentage points
- This suggests that having multiple descriptions (L3) provides some redundancy that compensates for brevity

### 3. Context Matters
- Longer descriptions include important contextual details:
  - Purpose/motivation ("to enable transactions")
  - Scope ("at various merchants")
  - Specific objects ("funds or benefits")
- These details help distinguish between semantically similar labels

### 4. Precision vs Recall Trade-off
- Both precision and recall decreased with shorter descriptions
- The model became less confident overall (though confidence metrics remained similar)

## Recommendations

### ❌ Do NOT Use Optimized (10-15 word) Prompts
The shorter descriptions sacrifice too much semantic information, leading to measurable performance degradation.

### ✅ Keep Original (15-20 word) Prompts
The original prompts provide a better balance between:
- Conciseness (not overly verbose)
- Context (enough detail to distinguish labels)
- Performance (3% higher accuracy)

### 🔍 Alternative Approaches to Explore

1. **Task-Aware Descriptions**: Instead of making descriptions shorter, make them more *relevant* to the specific classification task
2. **Hybrid Approach**: Use 12-15 words (middle ground) with focus on discriminative features
3. **Quality over Quantity**: Focus on improving the *content* of descriptions rather than reducing length

## Conclusion

The experiment demonstrates that **description length is not the primary issue** with current performance. The 15-20 word descriptions already provide good semantic information. Future optimization efforts should focus on:

1. **Task-aware generation**: Descriptions that highlight differences between similar labels
2. **Discriminative features**: Emphasize what makes each label unique
3. **Semantic richness**: Maintain contextual details that aid classification

The 3% accuracy drop from shorter descriptions suggests that the semantic information in the additional 6-7 words is valuable for the model's decision-making process.

---

**Generated**: 2025-01-XX
**Dataset**: Banking77 (1000 test samples, 77 classes)
**Model**: sentence-transformers/all-mpnet-base-v2
**Seed**: 42
