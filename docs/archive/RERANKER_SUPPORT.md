# Reranker (Cross-Encoder) Desteği ✅

## 📋 Özet

Sistem artık cross-encoder reranker modellerini destekliyor. Reranker'lar bi-encoder'lardan daha yavaş ama daha doğru sonuçlar verir.

## 🎯 Desteklenen Reranker Modelleri

1. **BAAI/bge-reranker-v2-m3** (568M) - Multilingual
2. **jinaai/jina-reranker-v2-base-multilingual** (278M) - Multilingual
3. **Qwen/Qwen3-Reranker-0.6B** (600M) - Multilingual

## 🔧 Teknik Detaylar

### Reranker vs Bi-encoder

| Özellik | Bi-encoder | Reranker (Cross-encoder) |
|---------|------------|--------------------------|
| **Hız** | Çok hızlı | Yavaş |
| **Doğruluk** | İyi | Çok iyi |
| **Kullanım** | Büyük ölçekli | Küçük-orta ölçekli |
| **Encoding** | Ayrı ayrı | Birlikte |

### Nasıl Çalışır?

**Bi-encoder:**
```
Text → Embedding
Label → Embedding
Similarity = cosine(text_emb, label_emb)
```

**Reranker:**
```
[Text, Label] → Cross-encoder → Score
```

Reranker text ve label'ı birlikte işler, bu yüzden daha doğru ama daha yavaş.

## 📁 Yeni Dosyalar

### Pipeline
- `src/pipeline_reranker.py` - Reranker pipeline implementasyonu

### Notebook
- `notebooks/RERANKER_EXPERIMENTS.ipynb` - 3 reranker × 6 dataset = 18 deney

### Configs
- `experiments/reranker/*.yaml` - Reranker experiment config'leri

## 🚀 Kullanım

### Config Formatı

```yaml
experiment_name: ag_news_bge_reranker

dataset:
  name: ag_news
  split: test
  text_column: text
  label_column: label
  max_samples: 1000

task:
  type: zero_shot_classification
  label_mode: description
  language: en

models:
  reranker:  # Bi-encoder yerine reranker
    provider: hf
    name: BAAI/bge-reranker-v2-m3

pipeline:
  mode: reranker  # Mode: reranker

evaluation:
  metrics:
    - accuracy
    - macro_f1
    - per_class_f1

output:
  save_predictions: true
  save_metrics: true
  output_dir: results/raw
```

### Tek Deney Çalıştırma

```bash
python main.py --config experiments/reranker/ag_news_bge_reranker.yaml
```

### Notebook Kullanımı

Colab'da `notebooks/RERANKER_EXPERIMENTS.ipynb` açın:
- 18 deney otomatik oluşturulur
- Tüm deneyler çalıştırılır
- Sonuçlar Drive'a kaydedilir
- Grafikler ve tablolar oluşturulur

## 📊 Beklenen Performans

Reranker'lar genellikle bi-encoder'lardan %5-15 daha yüksek F1 skoru verir, özellikle:
- Çok sınıflı problemlerde (Banking77, 20 Newsgroups)
- Benzer sınıfların olduğu durumlarda
- Nüanslı metinlerde

## ⚠️ Önemli Notlar

1. **Hız**: Reranker'lar çok daha yavaş
   - Bi-encoder: ~1000 text/saniye
   - Reranker: ~10-50 text/saniye

2. **Bellek**: Daha fazla GPU belleği gerektirir

3. **Kullanım Senaryosu**:
   - Reranker: Yüksek doğruluk gerekli, küçük dataset
   - Bi-encoder: Hız gerekli, büyük dataset

## 🎯 Sonuç

Sistem artık hem bi-encoder hem de reranker destekliyor:
- **7 bi-encoder** × 6 dataset = 42 deney
- **3 reranker** × 6 dataset = 18 deney
- **Toplam**: 60 deney

Her iki yaklaşımın avantaj ve dezavantajlarını karşılaştırabilirsiniz!
