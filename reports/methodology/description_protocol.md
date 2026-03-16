# Label Description Generation Protocol

## Overview

To mitigate the risk of researcher-introduced bias in label descriptions, all L2
(single-description) and L3 (multi-description) label texts used in this study were
generated through a standardized, automated protocol rather than written manually.
This section documents the models, prompt template, decoding parameters, and
authoritative source mapping that constitute the protocol.

---

## Language Models

Two large language models were employed to produce two independent description sets,
enabling a robustness check across description sources (see Section on Multi-Source
Robustness).

| Set | Model | Provider | Role |
|-----|-------|----------|------|
| Set A | GPT-4o (`openai/gpt-4o`) | OpenAI | Primary description set |
| Set B | Claude 3.5 Sonnet (`anthropic/claude-3-5-sonnet`) | Anthropic | Independent replication set |

Both models were accessed via their respective APIs under identical generation
conditions to ensure comparability.

---

## Prompt Template

A single, fixed prompt template was applied uniformly across all datasets and labels:

```
Define the following text classification label in 15-20 words, focusing only on its
semantic core without using the label name itself.
Dataset: [Dataset Name]. Label: [Label Name].
```

The template was not modified for any individual dataset or label. The constraint
"without using the label name itself" was included to prevent circular definitions
and to encourage descriptions that capture the underlying semantic concept rather
than paraphrasing the label string.

---

## Decoding Parameters

All API calls were made with `temperature = 0`.

Setting temperature to zero forces the model to select the highest-probability token
at each decoding step, producing deterministic output. This ensures that the
descriptions are fully reproducible: re-running the generation script with the same
model version and API endpoint yields identical text. Determinism is a prerequisite
for the reproducibility standard expected in empirical NLP research.

---

## Authoritative Source Mapping

Where an established external reference exists for a dataset's label taxonomy, that
source was designated as the authoritative grounding for the generated descriptions.
The LLM was expected to produce descriptions consistent with the authoritative
definition; if the authoritative source was inaccessible at generation time, the
model's parametric knowledge served as a fallback (`llm_fallback`).

| Dataset | Authoritative Source | Source Type |
|---------|---------------------|-------------|
| `ag_news` | Wikipedia (first sentence, L2); Wikidata definitions (L3) | `wikipedia` |
| `yahoo_answers_topics` | Wikipedia (first sentence, L2); Wikidata definitions (L3) | `wikipedia` |
| `SetFit/20_newsgroups` | Wikipedia (first sentence, L2); Wikidata definitions (L3) | `wikipedia` |
| `dbpedia_14` | DBpedia Ontology class definitions | `dbpedia_ontology` |
| `banking77` | Official `categories.json` / dataset documentation | `dataset_documentation` |
| `imdb` | Psychology dictionary / aspect-based sentiment definitions | `psychology_dictionary` |
| `sst2` | Psychology dictionary / aspect-based sentiment definitions | `psychology_dictionary` |
| `zeroshot/twitter-financial-news-sentiment` | Psychology dictionary / aspect-based sentiment definitions | `psychology_dictionary` |
| `go_emotions` | Ekman (1992) or Plutchik (1980) emotion theory definitions | `ekman_theory` |

If no authoritative source was reachable for a given label, the generation was
recorded with `source_type: llm_fallback` in the provenance log.

---

## Provenance and Reproducibility

Full provenance metadata is stored alongside the generated descriptions:

- **`src/label_descriptions/generation_metadata.json`** — records the model
  identifier, exact prompt template, temperature value, and UTC timestamp for each
  generation run.
- **`src/label_descriptions/provenance.json`** — contains one record per
  (dataset, label, label\_mode) triple, with fields `dataset`, `label_id`,
  `label_mode`, `source_type`, `source_url_or_reference`, and `generated_at`.

These files allow independent verification of every description used in the
experiments and support exact replication of the generation procedure.
