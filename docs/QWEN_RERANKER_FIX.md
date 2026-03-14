# Reranker Model Compatibility Issues

## Working Models
1. **BAAI/bge-reranker-v2-m3** (568M) ✅
2. **tomaarsen/Qwen3-Reranker-0.6B-seq-cls** (600M) ✅

## Excluded Models

### Jina Reranker v2 ❌
**Model**: jinaai/jina-reranker-v2-base-multilingual

**Error**:
```
ImportError: cannot import name 'create_position_ids_from_input_ids' from 'transformers.models.xlm_roberta.modeling_xlm_roberta'
```

**Root Cause**: 
- Jina reranker uses custom XLM-RoBERTa implementation
- Requires specific transformers version that conflicts with other dependencies
- The model's custom code imports deprecated functions from transformers

**Status**: Excluded from benchmark due to compatibility issues

## Benchmark Configuration
- Total experiments: 6 datasets × 2 rerankers = 12 experiments
- Both working models are multilingual and support 100+ languages
- Performance expected: 75-85% accuracy on most datasets

## Alternative Solutions (Not Implemented)
1. Use older transformers version (breaks other models)
2. Use Jina's official API (requires API key, not suitable for benchmark)
3. Wait for Jina to release updated model compatible with latest transformers
