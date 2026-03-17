# Prompt Optimization Testing - Key Takeaways

## What We Tested
Reduced label description length from 15-20 words to 10-15 words to test if shorter, more concise descriptions would improve performance.

## Results
❌ **Shorter descriptions decreased performance**

| Metric | Old (15-20 words) | New (10-15 words) | Change |
|--------|-------------------|-------------------|--------|
| L2 Accuracy | 60.5% | 57.5% | **-3.0%** |
| L3 Accuracy | 57.2% | 56.3% | **-0.9%** |
| Avg Word Count | 16.7 | 10.3 | -38% |

## Why Shorter Failed

### Example: Label 0 (activate_my_card)

**Old (15 words):**
> The user wants to activate their card to enable transactions and access their account services.

**New (8 words):**
> The user wants to activate their bank card.

**Lost Context:**
- Purpose: "to enable transactions"
- Scope: "access their account services"
- These details help distinguish from similar labels

## Key Insights

1. **Context > Brevity**: The extra 6-7 words provide valuable semantic information
2. **L2 more sensitive**: Single descriptions need more context than multiple descriptions (L3)
3. **Discriminative details matter**: Phrases like "at various merchants" or "funds or benefits" help distinguish similar labels

## Recommendation

✅ **Keep 15-20 word descriptions** - They provide the right balance of conciseness and context

🔍 **Next Steps**: Focus on task-aware descriptions that emphasize discriminative features, not shorter descriptions

## Files
- Full report: `results/optimized_prompts_comparison_report.md`
- Original descriptions restored: `src/label_descriptions/generated_descriptions.json`
- Backup of optimized: `src/label_descriptions/generated_descriptions_v1_original.json` (contains the old longer descriptions)
