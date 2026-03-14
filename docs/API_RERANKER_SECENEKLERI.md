# ⚡ API Tabanlı Reranker'lar vs Lokal Reranker

## 🐌 Neden Bu Kadar Yavaş?

**Sizin durum:** CPU'da cross-encoder çalıştırıyorsunuz

**Sorunlar:**
1. **CPU çok yavaş** - GPU 10-20x daha hızlı
2. **Her text-label çifti için model inference** - 1000 text × 3 candidates = 3000 inference
3. **Cross-encoder ağır** - BERT-style model her çift için

**Tipik hızlar:**
- GPU: ~2-3 dakika
- CPU: ~10-15 dakika
- API: ~30 saniye - 2 dakika

## 💰 API Tabanlı Reranker Seçenekleri

### 1. Cohere Rerank API ⭐ (En Popüler)

**Avantajlar:**
- ✅ Çok hızlı (API çağrısı)
- ✅ Güçlü model (multilingual)
- ✅ Kolay entegrasyon
- ✅ İlk 1000 request ücretsiz

**Dezavantajlar:**
- 💰 Ücretli (1000 sonrası)
- 🌐 İnternet gerekli
- 📊 Reproducibility zor

**Kullanım:**
```python
import cohere
co = cohere.Client("YOUR_API_KEY")

results = co.rerank(
    query="text to classify",
    documents=["label 1", "label 2", "label 3"],
    model="rerank-multilingual-v3.0"
)
```

**Fiyat:** $1 per 1000 requests
- 1000 text × 3 labels = 3000 requests = ~$3

### 2. Jina AI Reranker API

**Avantajlar:**
- ✅ Çok hızlı
- ✅ Multilingual
- ✅ Free tier var (10K requests/month)

**Dezavantajlar:**
- 💰 Sonrası ücretli
- 🌐 İnternet gerekli

**Kullanım:**
```python
import requests

headers = {"Authorization": "Bearer YOUR_API_KEY"}
data = {
    "model": "jina-reranker-v2-base-multilingual",
    "query": "text",
    "documents": ["label1", "label2"]
}

response = requests.post(
    "https://api.jina.ai/v1/rerank",
    headers=headers,
    json=data
)
```

**Fiyat:** Free 10K/month, sonra $0.002/1K requests

### 3. Voyage AI Reranker

**Avantajlar:**
- ✅ Çok güçlü
- ✅ Hızlı

**Dezavantajlar:**
- 💰 Daha pahalı
- 📝 Kayıt gerekli

**Fiyat:** $2.50 per 1M tokens

### 4. OpenAI Embeddings + Similarity (Alternatif)

**Not:** OpenAI reranker yok ama embeddings hızlı

```python
# Bi-encoder gibi kullan
embeddings = openai.Embedding.create(
    model="text-embedding-3-large",
    input=["text1", "text2"]
)
```

**Fiyat:** $0.13 per 1M tokens

## 🎯 Önerim: Sizin İçin En İyi Çözümler

### Seçenek 1: Google Colab (ÖNERİLEN - Ücretsiz) 🏆

**Neden en iyi:**
- ✅ Ücretsiz GPU
- ✅ Reproducible
- ✅ API key gerekmez
- ✅ Makale için daha iyi

**Hız:**
- CPU: ~10-15 dakika
- Colab GPU: ~2-3 dakika **5x daha hızlı!**

**Nasıl:**
1. Projeyi Drive'a yükle
2. `notebooks/01_run_experiments.ipynb` aç
3. Runtime → GPU
4. Çalıştır

### Seçenek 2: Daha Az Sample (HIZLI)

```yaml
# Config'de:
dataset:
  max_samples: 200  # 1000 yerine
```

**Hız:**
- 1000 sample: ~10 dakika
- 200 sample: ~2 dakika

**Not:** Makale için sonra tam veri ile çalıştırabilirsiniz.

### Seçenek 3: Sadece Bi-encoder (EN HIZLI)

```bash
# Hybrid yerine:
python main.py --config experiments/exp_agnews_baseline.yaml
```

**Hız:**
- Bi-encoder: ~1-2 dakika
- Hybrid: ~10-15 dakika

**Trade-off:** 2-3% daha düşük F1, ama çok hızlı

### Seçenek 4: Cohere API (Ücretli ama Hızlı)

**Karar:**
- **Araştırma/makale için:** Colab GPU kullan (ücretsiz, reproducible)
- **Production için:** API kullan (hızlı, sürdürülebilir)

## 💡 Sizin İçin En İyi Strateji

### Kısa Vadede (Bugün):

1. **Daha az sample ile test:**
```bash
# 200 sample ile hızlı test
python main.py --config experiments/exp_agnews_bge_reranker.yaml
# Config'de max_samples: 200 yapın
```

2. **Sadece bi-encoder deneyleri:**
```bash
# Çok hızlı
python main.py --config experiments/exp_agnews_name_only.yaml
python main.py --config experiments/exp_agnews_jina_biencoder.yaml
```

### Orta Vadede (Bu Hafta):

**Colab'a geç:**
- GPU ile hybrid deneyler çok hızlı
- Ücretsiz
- Makale için ideal

### Uzun Vadede (Production):

**API kullan:**
- Cohere Rerank ($3-5)
- Jina AI (free tier)
- Daha hızlı, sürdürülebilir

## 📊 Maliyet Karşılaştırması

| Çözüm | Maliyet | Hız | Reproducibility | Önerim |
|-------|---------|-----|-----------------|---------|
| **Colab GPU** | $0 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🏆 EN İYİ |
| **Az sample** | $0 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ İyi |
| **Bi-encoder only** | $0 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ İyi |
| **Cohere API** | $3-5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰 Ücretli |
| **Jina API** | $0-2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰 Free tier |

## 🚀 Hemen Yapılabilecekler

### 1. Hızlı Test (2 dakika)
```yaml
# experiments/exp_agnews_bge_reranker.yaml düzenle:
dataset:
  max_samples: 200
```

Sonra:
```bash
python main.py --config experiments/exp_agnews_bge_reranker.yaml
```

### 2. Bi-encoder Deneyleri (5 dakika toplam)
```bash
python main.py --config experiments/exp_agnews_name_only.yaml
python main.py --config experiments/exp_agnews_baseline.yaml
```

### 3. Colab'a Geç (30 dakika setup)
- Drive'a yükle
- GPU seç
- Hybrid deneyleri çok hızlı çalıştır

## 📝 Makale İçin

**En iyi yaklaşım:**

1. **Development:** Küçük sample + bi-encoder (hızlı iterasyon)
2. **Final deneyler:** Colab GPU (reproducible, ücretsiz)
3. **Makale:** Lokal model kullandık (API kullanmadık)

**Neden API kullanmamak daha iyi:**
- ✅ Reproducibility
- ✅ Okuyucular aynı sonuçları alabilir
- ✅ Bağımlılık yok
- ✅ Akademik standart

## ✅ Sonuç

**API kullanabilirsiniz ama şart değil!**

**En iyi çözüm sırasıyla:**
1. 🏆 **Colab GPU** (ücretsiz, hızlı, reproducible)
2. ✅ **Az sample** (hızlı test için)
3. ✅ **Bi-encoder only** (çok hızlı, yeterince iyi)
4. 💰 **API** (son çare, ücretli)

**Önerim:** Colab GPU'ya geçin, sorun çözülür! 🚀