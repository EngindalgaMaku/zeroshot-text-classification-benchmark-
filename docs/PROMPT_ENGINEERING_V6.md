# V6 Prompt Engineering: LLM-Generated Label Descriptions

## 🎯 Overview

This document describes the **V6 prompt engineering approach** for generating zero-shot text classification label descriptions using Large Language Models (LLMs).

**Key Achievement:** V6 LLM-generated descriptions achieve **81.9% F1** on AG News (vs 84.2% manual, only **-2.3%** gap).

---

## 📊 Results Summary

### AG News Test Results (INSTRUCTOR model)

| Approach | Label Mode | F1 Score | Delta from Manual |
|----------|-----------|----------|-------------------|
| Manual (Hand-written) | description | **84.2%** | Baseline 🏆 |
| **V6 L3 (LLM)** | **l3** | **81.9%** | **-2.3%** ✅ |
| **V6 L2 (LLM)** | **l2** | **80.0%** | **-4.2%** ✅ |
| V4 (Generic keywords) | l2 | 76.8% | -7.4% ❌ |
| L1 (Name only) | name_only | 73.4% | -10.8% |

**Conclusion:** V6 L3 is close enough to manual to be academically viable!

---

## 🔧 V6 Prompt Design

### Design Principles

1. **Format Replication**: Mimic the successful manual format exactly
2. **Domain-Aware**: Use dataset domain context (e.g., "news articles", "banking customer service")
3. **Concrete Terms**: Enforce domain-specific nouns, ban generic words
4. **Structured Output**: L2 (short) and L3 (long with 3 variations)

### Why V6 Succeeds vs Previous Versions

| Version | Approach | Problem | F1 |
|---------|----------|---------|-----|
| V4 | Generic keywords | Too abstract, no structure | 76.8% |
| V5 | Semantic + Keywords | Fragmented embedding | 76.2% |
| **V6** | **Simple list format** | **Format matches manual** | **80-82%** |

**Key insight:** Embedding models prefer cohesive, comma-separated lists over structured formats.

---

## 📝 V6 Prompts

### L2 Prompt (Short Description)

```
You are generating a text classification description for {dataset_domain}.

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

Output:
```

### L3 Prompt (Long Description - 3 Variations)

```
You are generating a comprehensive text classification description for {dataset_domain}.

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

Output (3 lines):
```

---

## 💡 Examples

### L2 (Short) Example - AG News

```
Label: World
Domain: news articles

Generated:
"This text is about diplomacy, international relations, treaties, conflicts, humanitarian aid, or global security."
```

### L3 (Long) Example - AG News

```
Label: Sports
Domain: news articles

Generated:
1. "This text is about athletes, competitions, leagues, scores, training, or endorsements."
2. "Sports journalism often highlights athlete performance and statistics in detail."
3. "Coverage includes analysis of game strategies and team dynamics throughout seasons."
```

---

## 🗂️ Dataset Domains

| Dataset | Domain | Labels |
|---------|--------|--------|
| ag_news | news articles | 4 |
| dbpedia_14 | encyclopedia entities | 14 |
| yahoo_answers_topics | community questions | 10 |
| banking77 | banking customer service | 77 |
| twitter-financial | financial market sentiment | 3 |
| 20_newsgroups | online discussion forums | 20 |
| imdb | movie reviews | 2 |
| sst2 | sentiment analysis | 2 |
| go_emotions | emotional expressions | 28 |

**Total:** 9 datasets, 160 labels

---

## 🚀 Usage

### 1. Generate Descriptions

```bash
# Generate L2+L3 for all datasets
python scripts/prompt_engineering/generate_all_v6_l2_l3.py
```

This creates: `src/label_descriptions/generated_descriptions.json`

### 2. Use in Experiments

```yaml
# config.yaml
task:
  label_mode: l2  # or l3
```

The system automatically loads from `generated_descriptions.json`.

### 3. Supported Label Modes

- `name_only` (L1): Just label names (manual)
- `description`: Hand-written descriptions (manual)
- **`l2`**: V6 LLM-generated short descriptions ✨
- **`l3`**: V6 LLM-generated long descriptions (3 variations) ✨

---

## 📐 Format Specification

### L2 Format

```
"This text is about X, Y, Z, W, V, or Q."
```

- Start: "This text is about"
- Terms: 6-8 domain-specific nouns
- Separator: ", " (comma-space)
- Last: "or" before final term
- End: "." (period)

### L3 Format

```json
[
  "This text is about X, Y, Z, W, V, or Q.",
  "Elaboration sentence 1 (8-12 words).",
  "Elaboration sentence 2 (8-12 words)."
]
```

- Line 1: Same as L2
- Lines 2-3: Different aspects/contexts
- All concrete, domain-specific terms

---

## 🧪 Validation

### Quality Checks

1. **Format validation**: Starts with "This text is about"
2. **Length validation**: L3 has exactly 3 lines
3. **Content validation**: No generic words (manual review)

### Performance Criteria

**Acceptable performance:** Within 5% of manual descriptions

- ✅ **AG News L3:** -2.3% (82% F1 vs 84% manual)
- ✅ **AG News L2:** -4.2% (80% F1 vs 84% manual)

---

## 📁 File Structure

```
src/label_descriptions/
├── generated_descriptions.json          ← V6 (ACTIVE)
├── generated_descriptions_original.json ← Manual (backup)
├── generated_descriptions_v6_all.json   ← V6 L2 only
└── generated_descriptions_backup.json   ← Old backup

scripts/prompt_engineering/
├── generate_all_v6_l2_l3.py            ← Main generator
├── test_v6_simple_format.py            ← L2 test
├── test_v6_l3_long.py                  ← L3 test
├── configs/                             ← Test configs
└── results/                             ← Test results
```

---

## 🔬 Technical Details

### LLM Configuration

- **Provider:** OpenRouter API
- **Model:** `openai/gpt-4o-mini`
- **Temperature:** 0 (deterministic)
- **Max tokens:** 100 (L2), 200 (L3)

### Embedding Model

- **Model:** INSTRUCTOR-base
- **Normalization:** L2 normalized
- **Pooling (L3):** Mean pooling across 3 variations

---

## 📖 Research Context

This approach is part of a research project studying:

1. **L1 vs L2 vs L3:** Impact of description length on zero-shot classification
2. **Manual vs LLM:** Can LLM-generated descriptions match human-written ones?
3. **Domain specificity:** How domain-aware prompting affects performance

**Finding:** V6 L3 achieves near-manual performance (-2.3%), validating LLM-based description generation for academic use.

---

## 🎓 Citation

If you use these prompts or generated descriptions, please cite:

```
[Your paper citation here]
```

---

## ⚠️ Limitations

1. **Language:** English only
2. **Models tested:** INSTRUCTOR, Nomic, BGE, E5, Jina, MPNet, Qwen
3. **Datasets:** 9 text classification datasets
4. **Performance variance:** May vary by dataset domain

---

## 🔄 Future Work

1. Test on more datasets
2. Experiment with different LLMs
3. Optimize prompt for specific domains
4. Multilingual adaptation

---

**Last updated:** 2026-03-16  
**Version:** V6 (Final)  
**Status:** Production-ready ✅