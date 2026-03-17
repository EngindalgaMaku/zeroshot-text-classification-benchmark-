# LLM-Based Label Description Generation Report

## 1. Overview

This report describes the automated generation of label descriptions for zero-shot text classification using Large Language Models (LLMs). The system produces two types of descriptions:
- **L2**: Single semantic description per label
- **L3**: Three multi-aspect descriptions per label

## 2. LLM Configuration

| Parameter | Value |
|-----------|-------|
| **Provider** | OpenRouter (accessing multiple models) |
| **Model** | Default model via OpenRouter API |
| **Temperature** | 0 (deterministic generation) |
| **Retry Strategy** | Up to 3 retries with exponential backoff |

## 3. Task-Aware Prompt Templates

The system uses task-specific prompt templates defined in `prompt_templates.yaml`:

### Supported Task Types
- `topic`
- `entity` 
- `sentiment`
- `emotion`
- `intent`

### L2 Template Structure
```
You are generating an L2 label description for a {task_type} task. 
Write 1 sentence (12-25 words) that MUST include the label text verbatim: "{label_name}". 
{task_specific_instructions}. Dataset: {dataset_name}. Label: {label_name}.
```

### L3 Template Structure
```
You are generating an L3 label description for a {task_type} task. 
Return EXACTLY 3 numbered sentences (12-25 words each). 
Each sentence MUST include "{label_name}" and focus on different {aspects}. 
Dataset: {dataset_name}. Label: {label_name}.
```

## 4. Dataset-to-Task-Type Mapping

| Dataset | Task Type | Labels |
|---------|-----------|--------|
| `ag_news` | topic | 4 classes (World, Sports, Business, Sci/Tech) |
| `yahoo_answers_topics` | topic | 10 classes |
| `SetFit/20_newsgroups` | topic | 20 classes |
| `dbpedia_14` | entity | 14 classes |
| `banking77` | intent | 77 classes |
| `imdb` | sentiment | 2 classes (Positive/Negative) |
| `sst2` | sentiment | 2 classes (Positive/Negative) |
| `zeroshot/twitter-financial-news-sentiment` | sentiment | 3 classes |
| `go_emotions` | emotion | 28 classes |

## 5. Generated Description Examples

### AG News (Topic Classification)

**Label 0 (World) - L2:**
```
This text is about diplomacy, international relations, treaties, conflicts, humanitarian aid, or global security.
```

**Label 0 (World) - L3:**
```
1. This text is about geopolitics, international relations, diplomacy, global conflicts, humanitarian crises, or economic sanctions.
2. Global conflicts often reshape alliances and influence international trade dynamics.
3. Humanitarian crises highlight the urgent need for international aid and cooperation.
```

**Label 1 (Sports) - L2:**
```
This text is about athletes, championships, scores, leagues, tournaments, or records.
```

**Label 1 (Sports) - L3:**
```
1. This text is about athletes, competitions, leagues, scores, training, or endorsements.
2. Sports journalism often highlights athlete performance and statistics in detail.
3. Coverage includes analysis of game strategies and team dynamics throughout seasons.
```

### DBpedia-14 (Entity Classification)

**Label 0 (Company) - L2:**
```
This text is about corporation, enterprise, startup, conglomerate, partnership, or limited liability company.
```

**Label 0 (Company) - L3:**
```
1. This text is about corporations, enterprises, businesses, firms, organizations, or startups.
2. Companies engage in commercial activities to generate profit and provide goods.
3. Corporate structures often include shareholders, management, and operational divisions.
```

## 6. Validation Rules

The system applies strict validation rules to ensure quality:

### Content Validation
- **Label Inclusion**: Every description MUST contain the exact label text
- **Word Count**: 12-25 words per sentence
- **Format**: L3 must return exactly 3 numbered sentences

### Semantic Validation
- **Task Consistency**: Descriptions must match the task type (e.g., sentiment descriptions use polarity language)
- **Distinctness**: L3 sentences must focus on different aspects
- **Coherence**: Descriptions must be grammatically correct and meaningful

### Error Handling
- Individual label failures don't stop batch processing
- Failed generations are logged with error details
- Retry mechanism handles temporary API issues

## 7. Metadata and Provenance

Each generation operation captures comprehensive metadata:

```json
{
  "timestamp": "2025-01-XX:XX:XX",
  "dataset": "ag_news",
  "label": "0",
  "task_type": "topic",
  "mode": "l2",
  "prompt_template": "...",
  "llm_provider": "openrouter",
  "model": "...",
  "temperature": 0,
  "generation_time_seconds": 1.2,
  "success": true,
  "description": "...",
  "validation_passed": true
}
```

## 8. Quality Assurance

### Deterministic Generation
- Temperature set to 0 ensures reproducible results
- Same prompt + label = identical description across runs

### Task-Specific Framing
- Topic descriptions use themes/subject matter framing
- Sentiment descriptions use polarity/emotional tone
- Entity descriptions define types and typical mentions
- Intent descriptions describe user goals/requests
- Emotion descriptions describe feelings and expressions

### Structured Output
- L2: Single sentence, 12-25 words
- L3: Exactly 3 numbered sentences, 12-25 words each
- Consistent formatting across all datasets and labels

## 9. Usage in Experiments

Generated descriptions are automatically loaded when experiments use `label_mode: l2` or `label_mode: l3`:

```yaml
# Example experiment configuration
dataset: sst2
label_mode: l2  # or l3, or name_only
models:
  encoder: hkunlp/instructor-large
```

The system seamlessly integrates with existing zero-shot classification pipelines, providing enhanced semantic understanding compared to raw label names alone.
