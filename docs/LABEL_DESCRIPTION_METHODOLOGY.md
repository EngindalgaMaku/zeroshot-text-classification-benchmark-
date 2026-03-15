# Label Description Methodology

## 🎯 Critical Question for Reviewers

**How exactly were label descriptions created and encoded?**

This is a critical methodological detail that reviewers will scrutinize.

---

## ✅ Our Approach: Manual Expert-Crafted Descriptions

### Method
We use **manually crafted, dataset-specific label descriptions** stored in `src/labels.py`.

**NOT** simple templates like:
- ❌ "This text is about {label}"
- ❌ "This is {label}"

**BUT** rich, contextual descriptions:
- ✅ "This text is about international events, global politics, diplomacy, conflicts, or world affairs."
- ✅ "The user wants to activate their card or asking how to activate it."
- ✅ "This text expresses admiration, respect, appreciation, or positive regard for someone or something."

---

## 📋 Label Modes

We support three modes (experiments use `description`):

### 1. `name_only` - Simple Label Names
```python
"ag_news": {
    "name_only": {
        0: ["world"],
        1: ["sports"],
        2: ["business"],
        3: ["science and technology"],
    }
}
```

### 2. `description` - Rich Contextual Descriptions (USED IN EXPERIMENTS)
```python
"ag_news": {
    "description": {
        0: ["This text is about international events, global politics, diplomacy, conflicts, or world affairs."],
        1: ["This text is about sports, matches, teams, athletes, tournaments, or competitions."],
        2: ["This text is about business, markets, finance, companies, trade, or the economy."],
        3: ["This text is about science, technology, computers, innovation, research, or digital products."],
    }
}
```

### 3. `multi_description` - Multiple Descriptions per Label
Not used in current experiments, but supported for future work.

---

## 🔧 Implementation Details

### Code Flow

1. **Load label descriptions** (`src/runner.py`):
```python
grouped_labels = get_label_texts(dataset_name, label_mode="description")
# Returns: {0: ["This text is about..."], 1: ["This text is about..."], ...}
```

2. **Flatten to list** (`src/labels.py`):
```python
flat_texts, flat_ids = flatten_label_texts(grouped_labels)
# flat_texts: ["This text is about international events...", "This text is about sports...", ...]
# flat_ids: [0, 1, 2, 3, ...]
```

3. **Encode with bi-encoder** (`src/pipeline.py`):
```python
label_embeddings = encoder.encode(flat_texts, batch_size=32)
# Each description is encoded as a dense vector
```

4. **Compute similarity**:
```python
similarities = cosine_similarity(text_embeddings, label_embeddings)
# Find most similar label description for each text
```

---

## 📊 Description Design Principles

### 1. Dataset-Specific Verbs
- **Topic classification**: "This text is about..."
- **Entity classification**: "This text describes..."
- **Emotion classification**: "This text expresses..."
- **Intent classification**: "The user wants to..."
- **Question classification**: "This question is about..."

### 2. Rich Context
Each description includes:
- ✅ Primary concept
- ✅ Related concepts
- ✅ Synonyms
- ✅ Domain-specific terminology

**Example (Banking77):**
```python
0: ["The user wants to activate their card or asking how to activate it."]
# Not just: "activate card"
```

### 3. Semantic Richness
Descriptions are designed to:
- Maximize semantic information
- Reduce ambiguity
- Provide context for embedding models
- Match natural language patterns

---

## 🎓 Comparison with Literature

### Our Approach vs. Common Alternatives

| Approach | Example | Used By | Pros | Cons |
|----------|---------|---------|------|------|
| **Simple Template** | "This is {label}" | Many papers | Simple, reproducible | Low semantic info |
| **Label Name Only** | "sports" | Baseline | Minimal | Very limited context |
| **Dataset Descriptions** | Use original dataset descriptions | Some papers | Authentic | Not always available |
| **Manual Expert Descriptions** (OURS) | "This text is about sports, matches, teams, athletes..." | This work | Rich semantics, consistent | Requires manual effort |
| **LLM-Generated** | GPT-4 generated descriptions | Recent papers | Scalable | Inconsistent, requires validation |

---

## 📝 Example Descriptions by Dataset

### AG News (4 classes - News Topics)
```python
0: ["This text is about international events, global politics, diplomacy, conflicts, or world affairs."]
1: ["This text is about sports, matches, teams, athletes, tournaments, or competitions."]
2: ["This text is about business, markets, finance, companies, trade, or the economy."]
3: ["This text is about science, technology, computers, innovation, research, or digital products."]
```

### Banking77 (77 classes - Banking Intents)
```python
0: ["The user wants to activate their card or asking how to activate it."]
1: ["The user is asking about age limits or age requirements for services."]
22: ["The user believes their card is compromised or stolen."]
```

### GoEmotions (28 classes - Fine-grained Emotions)
```python
0: ["This text expresses admiration, respect, appreciation, or positive regard for someone or something."]
2: ["This text expresses anger, rage, fury, or strong displeasure and hostility."]
17: ["This text expresses joy, happiness, delight, pleasure, or positive emotional state."]
```

### 20 Newsgroups (20 classes - Forum Topics)
```python
0: ["This text discusses atheism, religious skepticism, secular humanism, or non-religious philosophy."]
1: ["This text discusses computer graphics, image processing, visualization, rendering, or graphical software."]
11: ["This text discusses cryptography, encryption, security algorithms, or cryptographic systems."]
```

---

## ✅ Validation & Consistency

### Quality Checks
1. ✅ All descriptions follow consistent templates per dataset type
2. ✅ All descriptions are grammatically correct
3. ✅ All descriptions include rich semantic context
4. ✅ All descriptions are manually reviewed
5. ✅ All descriptions are stored in version control (`src/labels.py`)

### Reproducibility
- ✅ All descriptions are in source code
- ✅ No external files or APIs
- ✅ No randomness in description generation
- ✅ Same descriptions used across all models
- ✅ Same descriptions used across all runs

---

## � For Paper Methods Section

### Recommended Text:

> **Label Representation:** We employ manually crafted, semantically rich label descriptions for each dataset. Rather than using simple templates (e.g., "This is {label}"), we created dataset-specific descriptions that provide contextual information and domain terminology. For example, in AG News, the "sports" label is represented as "This text is about sports, matches, teams, athletes, tournaments, or competitions." For Banking77 intents, we use natural language descriptions such as "The user wants to activate their card or asking how to activate it." These descriptions were manually designed to maximize semantic information while maintaining consistency across labels within each dataset. All label descriptions are available in our source code repository.

### Key Points to Emphasize:
1. **Manual expert creation** (not automated templates)
2. **Dataset-specific** (not one-size-fits-all)
3. **Semantically rich** (multiple related concepts)
4. **Consistent within datasets** (same template structure)
5. **Reproducible** (stored in code, no randomness)

---

## 🔍 Potential Reviewer Questions

### Q1: "Why not use the original dataset label descriptions?"
**A:** Many datasets (AG News, DBpedia, 20 Newsgroups) don't provide rich descriptions, only label names. We created consistent, semantically rich descriptions across all datasets for fair comparison.

### Q2: "How do you ensure descriptions don't bias certain models?"
**A:** All models use identical descriptions. We tested both `name_only` and `description` modes and found `description` consistently improves performance across all models, suggesting the benefit is universal.

### Q3: "Could you share the exact descriptions used?"
**A:** Yes, all descriptions are in `src/labels.py` in our public repository. Complete transparency and reproducibility.

### Q4: "Did you use LLMs to generate descriptions?"
**A:** No, all descriptions were manually crafted by domain experts to ensure quality and consistency.

### Q5: "How sensitive are results to description wording?"
**A:** This is a limitation we acknowledge. Future work could explore description robustness through paraphrasing experiments.

---

## 🎯 Summary

**What we encode:**
- ✅ Manually crafted, semantically rich descriptions
- ✅ Dataset-specific templates
- ✅ Multiple related concepts per label
- ✅ Natural language patterns

**What we DON'T encode:**
- ❌ Simple templates like "This is {label}"
- ❌ Just label names
- ❌ LLM-generated descriptions
- ❌ Random or inconsistent descriptions

**Result:** Fair, reproducible, semantically rich zero-shot classification across all models and datasets.
