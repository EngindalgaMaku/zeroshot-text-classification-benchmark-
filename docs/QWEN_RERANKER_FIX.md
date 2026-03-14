# Qwen Reranker Model Fix

## Problem
The original Qwen reranker model `Qwen/Qwen3-Reranker-0.6B` was failing to load with sentence-transformers CrossEncoder due to incompatible model architecture.

## Error
```
File "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/auto_factory.py", line 354, in from_pretrained
    model_class = get_class_from_dynamic_module(
File "/usr/local/lib/python3.12/dist-packages/transformers/dynamic_module_utils.py", line 583, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
```

## Root Cause
The original `Qwen/Qwen3-Reranker-0.6B` model uses a custom architecture (`AutoModelForCausalLM`) that is not compatible with sentence-transformers' CrossEncoder class, which expects `AutoModelForSequenceClassification`.

## Solution
Use the converted model: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`

This is a community-converted version that:
- Uses `AutoModelForSequenceClassification` architecture
- Works seamlessly with sentence-transformers CrossEncoder
- Maintains the same performance as the original model
- Supports the same multilingual capabilities (100+ languages)

## Model Details
- **Model**: tomaarsen/Qwen3-Reranker-0.6B-seq-cls
- **Parameters**: 600M
- **Architecture**: Sequence Classification (compatible with CrossEncoder)
- **Languages**: 100+ languages
- **Context Length**: 32k tokens
- **Performance**: Same as original Qwen3-Reranker-0.6B

## Usage
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
pairs = [["query", "document"]]
scores = model.predict(pairs)
```

## Benchmark Impact
Now all 3 rerankers work correctly:
1. BAAI/bge-reranker-v2-m3 (568M)
2. jinaai/jina-reranker-v2-base-multilingual (278M)
3. tomaarsen/Qwen3-Reranker-0.6B-seq-cls (600M)

Total experiments: 6 datasets × 3 rerankers = 18 experiments

## References
- Original model: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- Converted model: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
- Conversion discussion: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls#updated-usage
