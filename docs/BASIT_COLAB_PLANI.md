# 🎯 BASİT VE GERÇEKÇI PLAN

## ❌ YANLIŞ YAPTIKLAR

- ❌ GTE: RoPE hatası, özel encoder gerek
- ❌ Qwen3-4B: 8GB model, çok büyük
- ❌ Jina v3: API key veya sorunlu

## ✅ ZATEN ÇALIŞAN

```
BGE-M3 description: 77.09% ✅
BGE-M3 hybrid: 77.97% ✅
```

---

## 🚀 YENİ GERÇEKÇI PLAN

### Local Test (Hemen)

**Küçük ve kolay modeller:**

1. **all-mpnet-base-v2** (420MB)
```bash
python main.py --config experiments/exp_agnews_mpnet.yaml
```

2. **all-MiniLM-L6-v2** (90MB)  
```bash
python main.py --config experiments/exp_agnews_minilm.yaml
```

3. **Hybrid test**
```bash
python main.py --config experiments/exp_agnews_mpnet_hybrid.yaml
```

---

### Colab İçin (Sonra)

**Orta boy modeller (Colab GPU ile):**

1. GTE (özel encoder ile)
2. Qwen3-Embedding
3. Jina reranker

---

## 📊 HEDEFİMİZ

**Makale için yeterli:**

| Model | Pipeline | Macro F1 |
|-------|----------|----------|
| BGE-M3 | description | 77.09% |
| BGE-M3 | hybrid | 77.97% |
| **MPNet** | description | **??%** |
| **MiniLM** | description | **??%** |
| **MPNet** | hybrid | **??%** |

**3-4 model yeterli!**

---

## 🎓 MAKALE İÇİN YETERLI Mİ?

**EVET!** ✅

### Contributions:

1. ✅ Modern embedding comparison (BGE, MPNet)
2. ✅ Hybrid pipeline evaluation
3. ✅ Label mode ablation (name vs description)
4. ✅ Error analysis

### Research Question:

> "How effective are hybrid bi-encoder + reranker pipelines for zero-shot text classification compared to single-stage approaches?"

---

## 🔧 ŞİMDİ NE YAPACAĞIZ?

### Adım 1: MPNet config oluştur

```yaml
experiment_name: agnews_mpnet_description

models:
  biencoder:
    name: sentence-transformers/all-mpnet-base-v2

task:
  label_mode: description
```

### Adım 2: Çalıştır

```bash
python main.py --config experiments/exp_agnews_mpnet.yaml
```

**Beklenen:** 75-78% (hızlı, 2-3 dakika)

### Adım 3: Hybrid test

```bash
python main.py --config experiments/exp_agnews_mpnet_hybrid.yaml
```

---

## 💡 NEDEN BU DAHA İYİ?

✅ Küçük modeller → Hızlı indirme  
✅ SentenceTransformers native → Hata yok  
✅ Local çalışır → GPU gerekmiyor  
✅ 3-4 model → Makale için yeterli  
✅ Colab'a geçmeden bitirebiliriz  

---

## 📝 SONRAKI ADIMLAR

1. ✅ MPNet config oluştur
2. ▶️ MPNet deneyi çalıştır  
3. ▶️ MiniLM deneyi çalıştır
4. ▶️ Hybrid test et
5. 📊 Sonuçları karşılaştır
6. 📄 Makale tablosu hazırla

**Bu hafta bitebilir! 🎉**