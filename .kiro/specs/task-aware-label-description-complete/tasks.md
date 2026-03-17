# Implementation Plan: Label Description Generation & Evaluation

## Overview

This plan implements the frozen label description generation protocol defined in
`scripts/talimatlar/`. The research goal is to compare L1/L2/L3 label representations
for embedding-based zero-shot text classification across 9 datasets.

Frozen decisions (from talimatlar/README.md):
- Task-type-specific prompts (not dataset-specific)
- temperature=0, fixed model
- Label-anchored embeddings: L1=`{label_name}`, L2=`{label_name}: {desc}`, L3=mean-pool of `{label_name}: {desc_i}`
- No prompt iteration after freeze

## Phase 1: Config & Prompt Freeze

- [x] 1. Update prompt_templates.yaml to match talimatlar/prompt_templates.yaml
  - Replace src/label_descriptions/prompt_templates.yaml with the frozen prompts from scripts/talimatlar/prompt_templates.yaml
  - Add version field "final_v1"
  - Prompts must NOT include "This text is about..." framing — use clean definitional style
  - Prompts must NOT include dataset_name in template (talimatlar prompts only use label_name)

- [x] 2. Update dataset_task_types.yaml to match talimatlar/dataset_task_types.yaml
  - Replace src/label_descriptions/dataset_task_types.yaml with the frozen config from scripts/talimatlar/dataset_task_types.yaml
  - Ensure all 9 datasets are present with correct task_type

- [x] 3. Verify generate_label_descriptions.py uses only YAML prompts (no legacy constants)
  - Confirm PROMPT_L2 / PROMPT_L3 constants are removed (already done)
  - Confirm generate_multi() has no fallback to legacy constants (already done)
  - Confirm TaskAwareLabelGenerator is the only generation path

## Phase 2: Label Embedding Input Policy

- [x] 4. Implement label-anchored embedding in get_label_texts (src/labels.py)
  - L1 mode (name_only): return `{label_name}` — already works
  - L2 mode: return `{label_name}: {description}` — use l2_anchored logic (already implemented)
  - L3 mode: return list of `["{label_name}: {desc1}", "{label_name}: {desc2}", "{label_name}: {desc3}"]` — use l3_anchored logic (already implemented)
  - Make l2_anchored the default behavior for l2 mode (rename or alias)
  - Make l3_anchored the default behavior for l3 mode (rename or alias)
  - Keep old l2/l3 modes as non-anchored variants for ablation only

- [x] 5. Verify runner.py routes l3/l3_anchored through mean-pool path
  - Confirm `if label_mode in ("multi_description", "l3", "l3_anchored"):` is in place (already done)
  - Confirm l2/l2_anchored go through flatten path (single embedding per class)

## Phase 3: Description Generation (All 9 Datasets)

- [x] 6. Regenerate all descriptions with frozen prompts
  - Delete or archive existing generated_descriptions.json (old prompts, non-frozen)
  - Run generation for all 9 datasets with new frozen prompts:
    - `python scripts/generate_label_descriptions.py --dataset ag_news --level both`
    - `python scripts/generate_label_descriptions.py --dataset banking77 --level both`
    - `python scripts/generate_label_descriptions.py --dataset dbpedia_14 --level both`
    - `python scripts/generate_label_descriptions.py --dataset yahoo_answers_topics --level both`
    - `python scripts/generate_label_descriptions.py --dataset "SetFit/20_newsgroups" --level both`
    - `python scripts/generate_label_descriptions.py --dataset imdb --level both`
    - `python scripts/generate_label_descriptions.py --dataset sst2 --level both`
    - `python scripts/generate_label_descriptions.py --dataset "zeroshot/twitter-financial-news-sentiment" --level both`
    - `python scripts/generate_label_descriptions.py --dataset go_emotions --level both`
  - Verify all labels generated without errors
  - Save provenance and generation_metadata.json

## Phase 4: Experiment Configs (All 9 Datasets × All Modes)

- [x] 7. Create experiment configs for all datasets and modes
  - For each of the 9 datasets, create YAML configs for:
    - name_only (L1)
    - l2_anchored (L2 with label prefix)
    - l3_anchored (L3 with label prefix, mean-pooled)
  - Use sentence-transformers/all-mpnet-base-v2 as the evaluation model
  - max_samples: 1000, seed: 42
  - Datasets needing configs: ag_news, banking77, dbpedia_14, yahoo_answers_topics,
    SetFit/20_newsgroups, imdb, sst2, zeroshot/twitter-financial-news-sentiment, go_emotions

## Phase 5: Run Experiments & Collect Results

- [ ] 8. Run all experiments and collect results
  - For each dataset, run L1 / L2 / L3 experiments
  - Record Macro F1 for each
  - Compare against existing manual description baseline (description mode)
  - Target result table:

    | Dataset | L1 (name_only) | L2 (anchored) | L3 (anchored) | Manual |
    |---------|---------------|---------------|---------------|--------|
    | ag_news | ? | ? | ? | 83.1% |
    | banking77 | ? | ? | ? | 63.1% |
    | dbpedia_14 | ? | ? | ? | ? |
    | yahoo_answers_topics | ? | ? | ? | ? |
    | SetFit/20_newsgroups | ? | ? | ? | ? |
    | imdb | ? | ? | ? | ? |
    | sst2 | ? | ? | ? | ? |
    | twitter-financial | ? | ? | ? | ? |
    | go_emotions | ? | ? | ? | ? |

## Phase 6: Ablation — Anchoring Effect

- [ ] 9. Run anchoring ablation on ag_news and banking77
  - Compare l2 (no anchor) vs l2_anchored for both datasets
  - Compare l3 (no anchor) vs l3_anchored for both datasets
  - Document whether label anchoring consistently improves performance
  - ag_news results so far: l2=75.99%, l2_anchored=77.32% ✅

## Notes

- Do NOT iterate on prompts further — they are frozen at final_v1
- Do NOT use "This text is about..." framing in new prompts
- L2 and L3 modes always use label-anchored embeddings as default
- Old non-anchored l2/l3 modes kept only for ablation comparison
- DATASET_SOURCE_MAP is metadata only — all descriptions are LLM-generated
