# 🎯 Deney Planı - AG News Zero-Shot Classification

## ✅ Tamamlanan Deneyler

### 1. Baseline (TAMAM ✓)
```bash
python main.py --config experiments/exp_agnews_baseline.yaml
```
- **Model:** BAAI/bge-m3
- **Mode:** description
- **Pipeline:** bi-encoder only
- **Sonuç:** 77.1% Macro F1

## 📋 Önerilen Deney Sırası

### Aşama 1: Label Mode Karşılaştırması

#### 2. Name-Only (Minimal Baseline)
```bash
python main.py --config experiments/exp_agnews_name_only.yaml
```
- **Amaç:** En basit label ile performans
- **Beklenen:** 70-75% F1
- **Süre:** ~2 dakika

#### 3. Multi-Description (Zengin Labels)
```bash
# Config düzenle: exp_agnews_baseline.yaml
# label_mode: multi_description olarak değiştir
python main.py --config experiments/exp_agnews_baseline.yaml
```
- **Amaç:** Daha zengin label açıklamaları
- **Beklenen:** 78-80% F1
- **Süre:** ~2 dakika

---

### Aşama 2: Hybrid Pipeline (ÖNEMLİ!)

#### 4. BGE + Jina Reranker (Mevcut Hybrid)
```bash
python main.py --config experiments/exp_agnews_hybrid.yaml
```
- **Model:** BAAI/bge-m3 + jinaai/jina-reranker
- **Mode:** multi_description
- **Pipeline:** hybrid (top_k=3)
- **Beklenen:** 82-85% F1 ⭐
- **Süre:** ~5 dakika (reranker yavaş)

#### 5. BGE + BGE Reranker
```bash
python main.py --config experiments/exp_agnews_bge_reranker.yaml
```
- **Model:** BAAI/bge-m3 + BAAI/bge-reranker-v2-m3
- **Amaç:** Aynı aileden modeller daha uyumlu olabilir
- **Beklenen:** 83-86% F1
- **Süre:** ~5 dakika

---

### Aşama 3: Farklı Bi-Encoder Modelleri

#### 6. Jina Bi-encoder (Güçlü Model)
```bash
python main.py --config experiments/exp_agnews_jina_biencoder.yaml
```
- **Model:** jinaai/jina-embeddings-v3
- **Mode:** description
- **Pipeline:** bi-encoder only
- **Beklenen:** 79-82% F1
- **Süre:** ~3 dakika

#### 7. Jina + Jina Hybrid (En Güçlü Kombinasyon)
```bash
python main.py --config experiments/exp_agnews_jina_jina_hybrid.yaml
```
- **Model:** jinaai/jina-embeddings-v3 + jinaai/jina-reranker
- **Amaç:** En güçlü modeller birlikte
- **Beklenen:** 84-87% F1 🏆
- **Süre:** ~6 dakika

#### 8. MiniLM (Hafif ve Hızlı)
```bash
python main.py --config experiments/exp_agnews_minilm.yaml
```
- **Model:** sentence-transformers/all-mpnet-base-v2
- **Amaç:** Hız/performans trade-off
- **Beklenen:** 74-77% F1
- **Süre:** ~1 dakika (çok hızlı)

---

### Aşama 4: Ablation Study

#### 9. Top-K Değişimi (Hybrid için)
Config dosyasını kopyalayıp top_k değiştirin:
```yaml
pipeline:
  top_k: 5  # veya 10
```
- **Amaç:** Top-K değerinin etkisini görmek
- Deneyin: top_k = 1, 3, 5, 10

---

## 📊 Karşılaştırma Scripti

Tüm deneyleri çalıştırdıktan sonra:

```python
import json
import pandas as pd
from pathlib import Path

results = []
for f in sorted(Path("results/raw").glob("*_metrics.json")):
    with open(f) as fp:
        m = json.load(fp)
    results.append({
        "Deney": m.get("experiment_name", f.stem),
        "Bi-encoder": m.get("biencoder", "N/A")[:30],
        "Reranker": m.get("reranker", "None")[:30] if m.get("reranker") else "None",
        "Pipeline": m.get("pipeline_mode", "N/A"),
        "Label Mode": m.get("label_mode", "N/A"),
        "Accuracy": f"{m['accuracy']:.4f}",
        "Macro F1": f"{m['macro_f1']:.4f}",
        "Weighted F1": f"{m['weighted_f1']:.4f}",
    })

df = pd.DataFrame(results)
df = df.sort_values("Macro F1", ascending=False)
print("\n" + "="*80)
print("TÜM DENEY SONUÇLARI")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# En iyi 3
print("\n🏆 EN İYİ 3 SONUÇ:")
print(df.head(3)[["Deney", "Macro F1", "Pipeline"]].to_string(index=False))
```

---

## 🎯 Makale İçin Minimum Gereksinimler

### Temel Deneyler (Mutlaka):
- ✅ Baseline (description) - TAMAM
- ⏳ Name-only baseline
- ⏳ Multi-description
- ⏳ Hybrid pipeline (en az 1)

### İyileştirme Deneyleri (Önerilen):
- ⏳ Farklı bi-encoder modeli (Jina)
- ⏳ Farklı reranker (BGE reranker)
- ⏳ En iyi kombinasyon (Jina+Jina)

### Ablation Study (Opsiyonel ama İyi):
- ⏳ Top-K etkisi
- ⏳ Label mode karşılaştırması
- ⏳ Hız/performans trade-off

---

## ⏱️ Tahmini Süre

### Hızlı Plan (3 saat):
1. Baseline ✅
2. Hybrid (mevcut) - 5 dk
3. Name-only - 2 dk
4. Multi-description - 2 dk
5. Jina bi-encoder - 3 dk
**Toplam: ~12 dakika çalıştırma + analiz**

### Tam Plan (1 gün):
- Tüm 8 deney: ~30 dakika
- Analiz ve grafikler: 2-3 saat
- Makale tabloları: 1-2 saat

---

## 💡 İpuçları

1. **Sırayla ilerleyin:** Her deneyi bitirin, sonucu kaydedin
2. **Notlar tutun:** Hangi model ne yapıyor, not alın
3. **Grafik çizin:** Her grup deneyden sonra karşılaştırma yapın
4. **Hata analizi:** En iyi modelin hatalarına bakın
5. **Time tracking:** Her modelin süresini kaydedin (makale için)

---

## 🚀 Hemen Başlayın!

```bash
# En önemli deney - Hybrid pipeline:
python main.py --config experiments/exp_agnews_hybrid.yaml

# Sonra Jina+Jina (en güçlü):
python main.py --config experiments/exp_agnews_jina_jina_hybrid.yaml

# Karşılaştırma:
python compare_results.py  # (yukarıdaki scripti kullan)
```

**Başarılar!** 🎯