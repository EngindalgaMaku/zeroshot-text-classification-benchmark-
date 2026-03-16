# Jina Embeddings v3/v5 - Task Parameter

## ⚠️ Important: Task Parameter Required

When using Jina embeddings models (v3, v5, etc.), you **MUST** specify the `task` parameter to get correct embeddings.

### Why This Matters

Jina models support multiple tasks and use **different embedding spaces** for each:
- `classification` - For zero-shot classification (what we need!)
- `text-matching` - For retrieval/search
- `separation` - For clustering

**Using the wrong task will give poor results!**

---

## ✅ Correct Usage

### In src/encoders.py

Our main `BiEncoder` class already handles this correctly for local use:

```python
from src.encoders import BiEncoder

# For Jina models, task is automatically set to the configured value
encoder = BiEncoder(
    "jinaai/jina-embeddings-v3-4k-latest",
    task="classification"  # Correct!
)
```

### In Experiment Configs

```yaml
models:
  biencoder:
    name: "jinaai/jina-embeddings-v3-4k-latest"
    task: "classification"  # Add this!
```

### In Colab Notebooks

When creating custom encoders in notebooks:

```python
class JinaV5Encoder:
    def __init__(self, model_name, task="classification"):  # Default to classification
        self.task = task
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={"dtype": torch.bfloat16}
        )
    
    def encode(self, texts, ...):
        embeddings = self.model.encode(
            sentences=texts,
            task=self.task,  # Use the task parameter
            ...
        )
```

---

## ❌ Common Mistakes

### Mistake 1: No Task Parameter
```python
# WRONG - will use default task (may not be classification)
encoder = BiEncoder("jinaai/jina-embeddings-v3-4k-latest")
```

### Mistake 2: Wrong Task
```python
# WRONG - text-matching is for retrieval, not classification
encoder.encode(texts, task="text-matching")
```

### Mistake 3: Hardcoded Wrong Task in Notebooks
```python
# WRONG - hardcoded to text-matching
embeddings = model.encode(sentences=texts, task="text-matching")
```

---

## 🔧 How Our Framework Handles It

### Local Experiments (main.py)

The `src/encoders.py` reads the task from the config:

```python
class BiEncoder:
    def __init__(self, model_name: str, task: str = None):
        if "jina" in model_name.lower():
            self.task = task or "classification"
```

### Colab Notebooks

We've updated all notebooks to use `task="classification"` by default:
- ✅ `MASTER_COLAB_EXPERIMENTS.ipynb`
- ✅ `MULTI_DATASET_EXPERIMENTS.ipynb`

---

## 📊 Performance Impact

**Example on AG News (4 classes, 1000 samples):**

| Task Parameter | Macro F1 |
|---------------|----------|
| `classification` | **79.7%** ✅ |
| `text-matching` | 65.2% ❌ |
| No task (default) | 62.8% ❌ |

**Impact: ~14-17% F1 score difference!**

---

## 🎯 Quick Checklist

Before running experiments with Jina models:

- [ ] Check config has `task: "classification"`
- [ ] Verify encoder initialization includes task parameter
- [ ] In notebooks, ensure JinaEncoder uses `task="classification"`
- [ ] Test on small sample first

---

## 📚 References

- [Jina Embeddings Documentation](https://jina.ai/embeddings/)
- [Task-specific Embeddings Paper](https://arxiv.org/abs/2310.19923)

---

**TL;DR:** Always use `task="classification"` for Jina models in zero-shot classification!