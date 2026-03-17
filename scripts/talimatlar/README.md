# Final Label Generation Config Bundle

This bundle freezes the prompt policy for label description generation in the
embedding-based zero-shot text classification study.

## What this is for

These files standardize label description generation for:
- L2: one short definitional sentence
- L3: three short definitional sentences

The main research question is not prompt engineering. The prompts are only a
controlled protocol for manipulating label semantics reproducibly.

## Frozen decisions

1. Use task-type-specific prompts, not dataset-specific prompts.
2. Use one fixed LLM model and temperature=0.
3. Use label-anchored embeddings:
   - L1: `{label_name}`
   - L2: `{label_name}: {description}`
   - L3: for each generated sentence, `{label_name}: {description_i}`, then mean-pool.
4. Do not use long descriptions.
5. Do not use numbering in final generated text.
6. Do not keep iterating prompts after this freeze unless there is a clear bug.

## Text input policy

The dataset text and the label descriptions are different things.

- Dataset text = the input instance to classify
- L1/L2/L3 = the label representation to compare against

Example for AG News:
- text embedding input: `title + " " + description`
- label embedding input:
  - L1: `Business`
  - L2: `Business: description`
  - L3: mean of three embeddings from:
    - `Business: desc1`
    - `Business: desc2`
    - `Business: desc3`

## Recommended implementation notes

- Strip whitespace aggressively.
- If L3 output is malformed, retry once or fail explicitly.
- Save metadata:
  - model name
  - prompt template version
  - temperature
  - timestamp
- Save generated label descriptions to JSON.

## Files

- `prompt_templates.yaml`
- `dataset_task_types.yaml`
