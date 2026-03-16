# 🚀 Tüm Modelleri Çalıştırma Rehberi

## 📋 Hazır Config'ler

### A. NLP Encoders (Embedding-based) - HIZLI ✅

```python
# Colab'da:

# 1. DeBERTa (SOTA NLP)
!python main.py --config experiments/exp_agnews_deberta.yaml

# 2. RoBERTa Large
!python main.py --config experiments/exp_agnews_roberta.yaml

# 3. ELECTRA Large
!python main.py --config experiments/exp_agnews_electra.yaml

# 4. MPNet (zaten var)
!python main.py --config experiments/exp_agnews_mpnet.yaml
```

**Süre:** ~15-20 dakika (hepsi)
**Beklenen:** 76-80% F1

### B. LLM Prompting - YAVAŞ ⏱️

```python
# 1. Qwen2.5-3B
!python main.py --config experiments/exp_agnews_qwen_3b_llm.yaml

# 2. Phi-3-mini
!python main.py --config experiments/exp_agnews_phi3_llm.yaml
```

**Süre:** ~20-30 dakika (hepsi)
**Beklenen:** 80-85% F1
**Not:** İlk 100 sample ile test (yavaş olduğu için)

### C. Hybrid Pipeline (Mevcut)

```python
# BGE + BGE reranker
!python main.py --config experiments/exp_agnews_bge_reranker.yaml
```

**Süre:** ~5 dakika
**Sonuç:** 79.4% F1 ✅

## 🎯 Önerilen Çalıştırma Sırası

### Faz 1: Hızlı Encoders (Önce)

```python
configs = [
    "experiments/exp_agnews_deberta.yaml",
    "experiments/exp_agnews_roberta.yaml",
    "experiments/exp_agnews_electra.yaml",
    "experiments/exp_agnews_mpnet.yaml",
]

for config in configs:
    print(f"\n{'='*60}\nRunning: {config}\n{'='*60}")
    !python main.py --config {config}

# Sonuçları görüntüle
!python compare_results.py
```

**Toplam süre:** ~20 dakika
**Sonuç:** 4 encoder modeli karşılaştırması

### Faz 2: LLM'ler (Sonra)

```python
llm_configs = [
    "experiments/exp_agnews_qwen_3b_llm.yaml",
    "experiments/exp_agnews_phi3_llm.yaml",
]

for config in llm_configs:
    print(f"\n{'='*60}\nRunning: {config}\n{'='*60}")
    !python main.py --config {config}

# Tüm sonuçlar
!python compare_results.py
```

**Toplam süre:** ~30 dakika
**Sonuç:** Embedding vs LLM karşılaştırması

## 📊 Beklenen Final Sonuçlar

### Encoder Models:
| Model | F1 Score | Type |
|-------|----------|------|
| BGE-m3 | 77.1% ✅ | Embedding |
| DeBERTa-v3 | 78-80% | NLP-SOTA |
| RoBERTa-large | 77-79% | NLP-Proven |
| ELECTRA-large | 76-78% | NLP-Efficient |
| MPNet | 75-77% | General |

### Hybrid:
| Pipeline | F1 Score |
|----------|----------|
| BGE + BGE reranker | 79.4% ✅ |

### LLM:
| Model | F1 Score (100 samples) |
|-------|------------------------|
| Qwen2.5-3B | 80-85%? |
| Phi-3-mini | 81-84%? |

## 💡 İpuçları

### Memory Sorunları:
```python
# LLM'ler için küçük batch
max_samples: 100  # İlk test için

# Sonra artırabilirsiniz
max_samples: 500  # Daha uzun ama daha kesin
```

### Hata Durumunda:
```python
# Runtime'ı restart edin
# Runtime → Restart runtime

# GPU check
import torch
print(torch.cuda.is_available())
```

### Hız Optimizasyonu:
```python
# Encoder modeller paralel çalıştırılabilir
# Ama LLM'ler tek tek çalıştırın (memory)
```

## ✅ Sonuç

**Tüm deneyler bitince:**

```python
# Karşılaştırma
!python compare_results.py

# Analiz
# notebooks/02_error_analysis.ipynb aç
# notebooks/03_tables_and_plots.ipynb aç
```

**Makale için:**
- 5 encoder model
- 2 LLM model  
- 1 hybrid pipeline
- Label semantics analizi

**KAPSAMLI BİR ÇALIŞMA!** 🎯