# Troubleshooting Guide

Common issues and solutions for the Zero-Shot Text Classification project.

## Installation Issues

### "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install -r requirements.txt
```

If that doesn't work:
```bash
pip install sentence-transformers --upgrade
```

### "CUDA out of memory"

**Solution 1:** Reduce batch size
Edit `src/encoders.py` and `src/pipeline.py`, change `batch_size=32` to `batch_size=8`

**Solution 2:** Reduce dataset size
In your config file:
```yaml
dataset:
  max_samples: 500  # Instead of 1000
```

**Solution 3:** Use CPU only
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### "SSL Certificate Verify Failed"

When downloading models from HuggingFace:

**Solution:**
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

Or:
```bash
pip install certifi
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

## Running Experiments

### "Dataset not found"

**For AG News:**
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")  # Will download automatically
```

If download fails, try:
```bash
export HF_DATASETS_CACHE="./data_cache"
```

### "Config file not found"

Make sure you're running from the project root:
```bash
cd /path/to/zero_shot_reliable_cls
python main.py --config experiments/exp_agnews_baseline.yaml
```

### Experiment is very slow

**On CPU:** Expected. Use Google Colab with GPU for faster execution.

**On GPU:** Check if GPU is actually being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

### "KeyError: 'text'" or "KeyError: 'label'"

Your dataset has different column names. Check with:
```python
from datasets import load_dataset
ds = load_dataset("your_dataset")
print(ds.column_names)
```

Then update your config:
```yaml
dataset:
  text_column: "your_text_column_name"
  label_column: "your_label_column_name"
```

## Google Colab Issues

### "Drive not mounting"

**Solution 1:** Clear cache and reconnect
```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

**Solution 2:** Check permissions
- Go to Google Drive
- Right-click on your folder
- Share → Make sure Colab has access

### "Runtime disconnected"

**Solution:** Enable longer runtime
- Runtime → Change runtime type
- Set GPU
- Consider Colab Pro for longer sessions

**Workaround:** Save frequently
All results are auto-saved to `results/raw/`, so you won't lose experiment data.

### "Package installation fails in Colab"

**Solution:**
```python
!pip install --upgrade pip
!pip install -r requirements.txt --no-cache-dir
```

### "Permission denied" when saving files

Make sure you're writing to Drive:
```bash
%cd /content/drive/MyDrive/zero_shot_reliable_cls
```

Not to local Colab storage:
```bash
# Don't do this:
%cd /content/zero_shot_reliable_cls
```

## Results & Analysis

### "No metrics file found"

Check if experiment finished successfully:
```bash
ls results/raw/
```

If empty, check logs for errors during experiment.

### "Predictions file is empty"

Make sure config has:
```yaml
output:
  save_predictions: true
  save_metrics: true
```

### "Plots not showing in notebook"

Add this at the top of notebook:
```python
import matplotlib.pyplot as plt
%matplotlib inline
```

### LaTeX tables not rendering

Install required packages:
```bash
pip install jinja2
```

Or just use CSV output:
```python
df.to_csv("results/tables/my_table.csv")
```

## Model Issues

### "Model download is stuck"

**Solution 1:** Set timeout
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("model_name", cache_folder="./models")
```

**Solution 2:** Download manually
```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-m3 models/bge-m3
```

Then update config to use local path:
```yaml
models:
  biencoder:
    name: ./models/bge-m3
```

### "Model gives poor results"

**Checklist:**
1. Is label mode appropriate? Try `description` instead of `name_only`
2. Are embeddings normalized? Should be `true` for cosine similarity
3. Is dataset clean? Check for label noise
4. Is the model suitable? Some models are better for certain domains

### "Reranker is slower than expected"

This is normal. Rerankers process each text-label pair individually.

**Solutions:**
- Use hybrid pipeline with `top_k=3` instead of exhaustive reranking
- Reduce dataset size for testing
- Use a smaller/faster reranker model

## Performance Issues

### Low accuracy on custom dataset

**Common causes:**
1. **Poor label descriptions**
   - Make labels more descriptive
   - Add multiple paraphrases
   
2. **Domain mismatch**
   - Pre-trained models may not know your domain
   - Try domain-specific models
   
3. **Label ambiguity**
   - Classes too similar
   - Consider merging similar classes

4. **Data quality**
   - Check for mislabeled examples
   - Remove duplicates
   - Ensure balanced classes

### High confidence but wrong predictions

This is interesting for analysis! These are the most important errors.

Check:
```python
errors = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
print(errors.head(20))
```

Common patterns:
- Borderline cases between similar classes
- Misleading keywords
- Complex or ambiguous texts

## Python/Environment Issues

### "Import error" after updating packages

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### "Kernel died" in Jupyter

**Solution 1:** Reduce memory usage (smaller batch size, fewer samples)

**Solution 2:** Restart kernel
- Kernel → Restart & Clear Output

**Solution 3:** Close other notebooks

### Version conflicts

Create a fresh virtual environment:
```bash
python -m venv venv_zeroshot
source venv_zeroshot/bin/activate  # Linux/Mac
# or
venv_zeroshot\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Still Having Issues?

1. **Run the test script:**
   ```bash
   python test_setup.py
   ```

2. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Check available disk space:**
   Models and datasets need ~5GB

4. **Try a minimal example:**
   ```bash
   python main.py --config experiments/exp_agnews_name_only.yaml
   ```
   
   This uses minimal resources.

5. **Enable debug mode:**
   Add to top of main.py:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Quick Fixes

### Clear everything and start fresh

```bash
# Remove cached files
rm -rf data_cache/*
rm -rf results/raw/*
rm -rf ~/.cache/huggingface/

# Reinstall packages
pip uninstall -y sentence-transformers transformers
pip install -r requirements.txt

# Run test
python test_setup.py
```

### Minimal working example

If nothing works, try this minimal script:

```python
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load data
dataset = load_dataset("ag_news", split="test[:100]")

# Encode
texts = dataset["text"]
embeddings = model.encode(texts)

print(f"Success! Encoded {len(embeddings)} texts")
```

If this works, the issue is in the project code, not the setup.

## Need More Help?

If you're still stuck:
1. Check error message carefully
2. Google the specific error
3. Check HuggingFace documentation
4. Review the example notebooks
5. Try with a different/simpler dataset first

Remember: Most issues are related to:
- Memory/GPU issues → Reduce batch size
- Package versions → Reinstall requirements
- Path issues → Check you're in the right directory
- Model downloads → Check internet connection