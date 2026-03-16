# Reranker Model Selection for Zero-Shot Classification

## Selected Models

We use 3 NLI (Natural Language Inference) cross-encoders specifically trained for zero-shot classification:

### 1. cross-encoder/nli-deberta-v3-large (435M params)
- **Architecture**: DeBERTa-v3-large
- **Training**: SNLI + MultiNLI datasets
- **Performance**: SOTA on NLI benchmarks
- **Use case**: Best accuracy, slower inference
- **Output**: 3 scores (contradiction, entailment, neutral)

### 2. cross-encoder/nli-roberta-base (125M params)
- **Architecture**: RoBERTa-base
- **Training**: SNLI + MultiNLI datasets
- **Performance**: Strong baseline
- **Use case**: Balanced speed/accuracy
- **Output**: 3 scores (contradiction, entailment, neutral)

### 3. MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (279M params)
- **Architecture**: mDeBERTa-v3-base (multilingual)
- **Training**: MNLI + XNLI (15 languages)
- **Performance**: Best multilingual support
- **Use case**: Cross-lingual zero-shot
- **Output**: 3 scores (contradiction, entailment, neutral)

## Why NLI Models?

NLI models are specifically designed for zero-shot classification because:

1. **Trained on premise-hypothesis pairs**: Perfect for text-label matching
2. **Entailment scoring**: Directly measures if text matches label
3. **Zero-shot transfer**: No task-specific fine-tuning needed
4. **Proven effectiveness**: Used in BART-MNLI, DeBERTa-MNLI pipelines

## Excluded Models

### BGE Reranker v2-m3 ❌
- **Reason**: Trained for retrieval, not NLI
- **Issue**: Not optimized for premise-hypothesis format
- **Performance**: Lower accuracy on zero-shot classification

### Qwen3 Reranker ❌
- **Reason**: Trained for retrieval/ranking tasks
- **Issue**: Requires specific input format, not NLI-based
- **Performance**: Not designed for zero-shot classification

### Jina Reranker v2 ❌
- **Reason**: Transformers compatibility issues
- **Error**: `ImportError: cannot import name 'create_position_ids_from_input_ids'`
- **Status**: Technical blocker

## Benchmark Configuration

- **Total experiments**: 6 datasets × 3 NLI models = 18 experiments
- **Input format**: NLI premise-hypothesis pairs
- **Expected accuracy**: 75-85% (based on literature)
- **Inference**: Slower than bi-encoders, more accurate

## Model Comparison

| Model | Params | Speed | Accuracy | Multilingual |
|-------|--------|-------|----------|--------------|
| NLI-DeBERTa-Large | 435M | Slow | Highest | No |
| NLI-RoBERTa-Base | 125M | Fast | Good | No |
| mDeBERTa-MNLI | 279M | Medium | High | Yes (15 langs) |

## References

1. **cross-encoder/nli-deberta-v3-large**: https://huggingface.co/cross-encoder/nli-deberta-v3-large
2. **cross-encoder/nli-roberta-base**: https://huggingface.co/cross-encoder/nli-roberta-base
3. **MoritzLaurer/mDeBERTa-v3-base-mnli-xnli**: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

## Usage

All models use the same NLI format:
```python
premise = "Apple announces new iPhone with advanced camera"
hypothesis = "This text is about technology."

# Model outputs entailment score
# High entailment = text matches label
```
