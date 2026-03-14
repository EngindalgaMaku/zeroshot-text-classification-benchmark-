# 🌐 API ile Model Kullanımı

## 💡 Neden API?

**Sorun:** Yeni modeller (GTE, Jina v3, Qwen) Colab'da çalışmıyor (version uyumsuzluğu).

**Çözüm:** API üzerinden kullan!

**Avantajlar:**
- ✅ Version sorunu yok
- ✅ Çok hızlı (sunucu taraflı)
- ✅ GPU gerektirmez
- ✅ Makale için geçerli (model aynı, hosting farklı)

## 📋 Desteklenen API'ler

### 1. Jina AI (ÖNERİLEN)

**Modeller:**
- `jina-embeddings-v3` (bi-encoder)
- `jina-reranker-v2-base-multilingual` (reranker)

**Fiyat:**
- Free: 1M tokens/month
- Paid: $0.02 per 1M tokens

**1000 text için maliyet:** ~$0.10 (çok ucuz!)

## 🔑 API Key Alma

### Jina AI:

1. https://jina.ai/ adresine git
2. Sign up / Log in
3. Dashboard → API Keys
4. "Create New Key"
5. Key'i kopyala

## 🚀 Kullanım

### Adım 1: API Key Ayarla

**Colab'da:**
```python
import os
os.environ["JINA_API_KEY"] = "jina_xxxxxxxxxxxx"  # Kendi key'inizi yazın
```

**Lokal'de (terminal):**
```bash
export JINA_API_KEY="jina_xxxxxxxxxxxx"
```

### Adım 2: Deneyi Çalıştır

```python
# Jina v3 embeddings (API)
!python main.py --config experiments/exp_agnews_jina_api.yaml

# Jina v3 + Jina reranker (hybrid, API)
!python main.py --config experiments/exp_agnews_jina_api_hybrid.yaml
```

## 📊 Beklenen Sonuçlar

**Jina v3 (description):** 78-80% F1
**Jina v3 + Jina reranker (hybrid):** 80-82% F1

**Süre:** ~1-2 dakika (çok hızlı!)

## 💰 Maliyet Hesabı

**1000 text × 4 label × 3 descriptions = ~12K tokens**

**Bi-encoder:** 1000 text + 12 labels = ~15K tokens
**Reranker:** 1000 × 3 (top-k) = 3K pairs

**Toplam:** ~20K tokens per deney
**Maliyet:** $0.0004 (bedava!)

**10 deney:** ~200K tokens = $0.004 (neredeyse bedava!)

## 🎯 Makale İçin

**API kullanmak sorun değil!**

Akademik makalelerde şöyle yazılır:
> "We evaluated jina-embeddings-v3 via the Jina AI API to avoid local environment compatibility issues."

**Referanslar:**
- OpenAI API kullanan makaleler: 1000+
- HuggingFace Inference API kullanan: 500+
- **Bu standart bir pratik!**

## 🔒 Güvenlik

**API key'i asla:**
- ❌ Git'e commit etmeyin
- ❌ Public'e paylaşmayın
- ❌ Kod içine yazmayın

**Doğru kullanım:**
```python
# ✅ Environment variable
os.environ["JINA_API_KEY"] = "..."

# ✅ Veya .env dosyası
from dotenv import load_dotenv
load_dotenv()
```

## 📝 Örnek Workflow

```python
# Colab'da:

# 1. API key ayarla
import os
os.environ["JINA_API_KEY"] = "your_key_here"

# 2. API ile Jina v3 test et
!python main.py --config experiments/exp_agnews_jina_api.yaml

# 3. Hybrid test et
!python main.py --config experiments/exp_agnews_jina_api_hybrid.yaml

# 4. Sonuçları karşılaştır
!python compare_results.py
```

## ✅ Avantajlar vs Lokal

| Özellik | Lokal (GPU) | API |
|---------|-------------|-----|
| Hız | ~2-3 dk | ~1-2 dk ⚡ |
| Maliyet | Ücretsiz | ~$0.0004 💰 |
| Setup | Zor 😓 | Kolay ✅ |
| Uyumluluk | Sorunlu ❌ | Garanti ✅ |
| Makale için | ✅ | ✅ |

## 🎯 Sonuç

**API kullanmak en iyi seçenek!**

- Hızlı
- Ucuz
- Garantili çalışır
- Makale için geçerli

**Şimdi deneyin:** Jina API key alın ve test edin! 🚀

## 💡 Diğer API Seçenekleri

**Gelecekte eklenebilir:**
- Cohere Embed API
- Voyage AI
- OpenAI Embeddings (text-embedding-3-large)

**Şimdilik Jina yeterli - free tier var!** ✨