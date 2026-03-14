# Batch Size Optimizasyonu 🚀

## 🐛 Sorun

Qwen gibi büyük modeller (4B-8B parametre) 20 Newsgroups gibi büyük datasetlerde memory overflow (OOM) hatası veriyor:

```
CUDA out of memory. Tried to allocate X GB
```

## 🔍 Sebep

- **Qwen3-Embedding-8B**: 8 milyar parametre
- **20 Newsgroups**: 5000 sample + 20 class
- **Default batch_size**: 32
- **Sonuç**: GPU belleği yetersiz

## ✅ Çözüm

### 1. Otomatik Batch Size Ayarlama

Runner'da model bazlı otomatik batch size:

```python
# Auto-detect based on model name
if any(x in name_lower for x in ["qwen", "8b", "7b", "13b"]):
    batch_size = 8  # Very large models
elif any(x in name_lower for x in ["large", "xl", "xxl"]):
    batch_size = 16  # Large models
else:
    batch_size = 32  # Default
```

### 2. Config'de Manuel Ayar

```yaml
pipeline:
  mode: biencoder
  normalize_embeddings: true
  batch_size: 4  # Manuel ayar (Qwen için)
```

### 3. Notebook'ta Otomatik

```python
# Qwen için küçük batch
batch_size = 8 if "qwen" in model_short.lower() else 32
```

## 📊 Önerilen Batch Size'lar

| Model Boyutu | Batch Size | Örnekler |
|--------------|------------|----------|
| Küçük (<500M) | 32-64 | MPNet, Jina v5 nano |
| Orta (500M-1B) | 16-32 | BGE-M3, E5-large, INSTRUCTOR |
| Büyük (1B-4B) | 8-16 | Qwen3-4B, Snowflake |
| Çok Büyük (>4B) | 4-8 | Qwen3-8B |

## 🎯 Güncellenen Dosyalar

### 1. Runner (`src/runner.py`)
- Otomatik batch size detection
- Model adına göre ayarlama
- Config'den override desteği

### 2. Qwen Configs
- `experiments/exp_20newsgroups_qwen.yaml`: batch_size=4
- `experiments/exp_agnews_qwen3.yaml`: batch_size=8

### 3. Notebook
- Qwen modelleri için otomatik batch_size=8

## ⚡ Performans Etkisi

Batch size küçültmenin etkileri:

| Batch Size | Hız | Bellek | Doğruluk |
|------------|-----|--------|----------|
| 32 | ⚡⚡⚡ | 🔥🔥🔥 | ✅ |
| 16 | ⚡⚡ | 🔥🔥 | ✅ |
| 8 | ⚡ | 🔥 | ✅ |
| 4 | 🐌 | ✓ | ✅ |

**Not**: Doğruluk etkilenmez, sadece hız azalır.

## 🔧 Kullanım

### Otomatik (Önerilen)

```bash
# Runner otomatik algılar
python main.py --config experiments/exp_20newsgroups_qwen.yaml
```

### Manuel Config

```yaml
pipeline:
  batch_size: 4  # Manuel ayar
```

### Notebook

```python
# Otomatik ayarlanır
# Qwen → batch_size=8
# Diğerleri → batch_size=32
```

## 🎉 Sonuç

Artık büyük modeller OOM hatası vermeden çalışacak:
- ✅ Qwen3-8B: batch_size=8 (otomatik)
- ✅ 20 Newsgroups: 5000 sample (sorunsuz)
- ✅ Memory overflow: Çözüldü

Hız biraz azalır ama model çalışır! 🚀
