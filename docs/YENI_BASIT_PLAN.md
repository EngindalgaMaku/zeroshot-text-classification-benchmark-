# 🎯 YENİ BASİT DENEY PLANI (Mart 2026)

## ✅ MEVCUT BAŞARILI SONUÇLAR

```
BGE-M3 + BGE reranker (hybrid): 77.97% Macro F1 ✅
BGE-M3 description (bi-encoder): 77.09% Macro F1 ✅
```

## ❌ BAŞARISIZ YAKLAŞIM

```
MLM (RoBERTa): 37.5% - SİLİNDİ ❌
```

---

## 🚀 YENİ MODELLER (Bu Hafta)

### 1️⃣ GTE-Large (Alibaba)

**Config:** `experiments/exp_agnews_gte.yaml`

```yaml
experiment_name: agnews_gte_description

models:
  biencoder:
    name: Alibaba-NLP/gte-large-en-v1.5

task:
  label_mode: description
```

**Komut:**
```bash
python main.py --config experiments/exp_agnews_gte.yaml
```

**Beklenen:** ~78-80% (BGE ile yarışmalı)

---

### 2️⃣ Qwen3-Embedding-4B

**Config:** `experiments/exp_agnews_qwen3_embedding.yaml`

```yaml
experiment_name: agnews_qwen3_embedding

models:
  biencoder:
    name: Qwen/Qwen3-Embedding-4B

task:
  label_mode: description
```

**Komut:**
```bash
python main.py --config experiments/exp_agnews_qwen3_embedding.yaml
```

**Beklenen:** ~78-82% (yeni nesil model)

---

### 3️⃣ GTE + BGE Reranker (Hybrid)

**Config:** `experiments/exp_agnews_gte_bge_hybrid.yaml`

```yaml
experiment_name: agnews_gte_bge_hybrid

models:
  biencoder:
    name: Alibaba-NLP/gte-large-en-v1.5
  reranker:
    name: BAAI/bge-reranker-v2-m3

pipeline:
  mode: hybrid
  top_k: 3

task:
  label_mode: multi_description
```

---

## 📊 DENEY SIRALAMASI

### Hafta 1 (Bu Hafta)

1. ✅ **GTE bi-encoder** (kolay, hızlı)
2. ✅ **Qwen3-Embedding bi-encoder** (yeni nesil test)
3. ✅ **GTE + BGE hybrid** (çapraz aile testi)

### Hafta 2 (Gelecek)

4. Qwen3-Embedding + Qwen3-Reranker (same-family)
5. Label mode ablation (name vs description vs multi)
6. Error analysis

---

## 🎓 MAKALEDEKİ YERİ

### Research Question

> "Modern embedding models ile zero-shot text classification'da hybrid pipeline (bi-encoder + reranker) ne kadar etkili?"

### Ana Tablo

| Model | Pipeline | Label Mode | Macro F1 |
|-------|----------|------------|----------|
| BGE-M3 | bi-encoder | description | 77.09% |
| BGE-M3 | hybrid | multi_desc | 77.97% |
| **GTE-large** | bi-encoder | description | **??%** |
| **Qwen3-4B** | bi-encoder | description | **??%** |
| **GTE** | hybrid | multi_desc | **??%** |

### Contributions

1. ✅ Modern embedding model karşılaştırması
2. ✅ Hybrid pipeline değerlendirmesi
3. ✅ Label engineering etkisi
4. ✅ Error analysis

---

## 🔧 TEKNİK DETAYLAR

### Kolay Modeller (sentence-transformers ile)

- BAAI/bge-m3 ✅ (zaten çalışıyor)
- Alibaba-NLP/gte-large-en-v1.5 ✅ (kolay)
- Qwen/Qwen3-Embedding-4B ✅ (kolay)

### Config Template

```yaml
experiment_name: agnews_{model_name}_{mode}

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
  biencoder:
    name: {MODEL_NAME}
  reranker: null  # veya reranker adı

pipeline:
  mode: biencoder  # veya hybrid
  normalize_embeddings: true

output:
  save_predictions: true
  save_metrics: true
  output_dir: results/raw
```

---

## 🗑️ SİLİNECEKLER

### MLM İle İlgili

- `src/approaches/mlm_classifier.py`
- `src/runner_mlm.py`
- `src/label_mappings.py`
- `experiments/mlm/` klasörü
- `test_mlm_*.py` dosyaları
- `results/mlm/` klasörü

**Neden?** 
- MLM zero-shot için değil
- 37.5% accuracy ile başarısız
- Kod karmaşık ve gereksiz

---

## ✅ KALACAKLAR

### Çalışan Sistem

- `src/encoders.py` ✅
- `src/rerankers.py` ✅
- `src/pipeline.py` ✅
- `src/runner.py` ✅
- `main.py` ✅
- `experiments/*.yaml` ✅

---

## 🎯 BU HAFTANIN HEDEFİ

**3 yeni deney sonucu:**

1. GTE bi-encoder → ?? %
2. Qwen3-Embedding → ?? %
3. GTE + BGE hybrid → ?? %

**Karşılaştırma tablosu:**

```
BGE-M3: 77.09% (baseline)
GTE: ?? %
Qwen3: ?? %
Hybrid: ?? %
```

**Target:** 80%+ Macro F1

---

## 📝 SONRAKI ADIMLAR

1. ✅ MLM dosyalarını sil
2. ✅ GTE config oluştur
3. ✅ Qwen3 config oluştur
4. ✅ Hybrid config oluştur
5. ▶️ Deneyleri çalıştır
6. 📊 Sonuçları karşılaştır

---

## 💪 NEDEN BU PLAN İYİ?

- ✅ Basit ve anlaşılır
- ✅ Çalışan kod üzerine kurulu
- ✅ Yeni modeller kolay entegre
- ✅ Makale için yeterli contribution
- ✅ Haftaya bitirebiliriz

**Başlayalım! 🚀**