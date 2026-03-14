# 🚀 Yeni Deney Planı - Güncel Modeller

## 🎯 Hedef

Modern embedding ve reranking modellerini sistematik olarak test etmek:
- SOTA bi-encoder'ları karşılaştırmak
- Same-family vs cross-family hybrid performansını ölçmek
- Hangi kombinasyonun en iyi olduğunu belirlemek

## 📋 Seçilen Modeller

### Bi-Encoders (4 model)

| Model | Aile | Parametre | Özellik |
|-------|------|-----------|---------|
| BAAI/bge-m3 ✅ | BGE | 567M | Multilingual, baseline |
| Alibaba-NLP/gte-large-en-v1.5 🆕 | GTE | 434M | English-optimized |
| jinaai/jina-embeddings-v3 🆕 | Jina | 570M | Task-adapted |
| Qwen/Qwen2.5-Embedding-1.5B 🆕 | Qwen | 1.5B | Largest, multilingual |

### Rerankers (2 model)

| Model | Aile | Özellik |
|-------|------|---------|
| BAAI/bge-reranker-v2-m3 ✅ | BGE | Baseline, stable |
| jinaai/jina-reranker-v2 🆕 | Jina | Latest version |

**Not:** Jina reranker v3 yerine v2 kullanıyoruz (Colab uyumluluk)

## 🔬 Deney Sırası

### Faz 1: Bi-encoder Karşılaştırması (4 deney)

**Amaç:** Hangi bi-encoder en iyi?

```bash
# 1. GTE (English-optimized)
!python main.py --config experiments/exp_agnews_gte_biencoder.yaml

# 2. Jina v3 (Task-adapted)
!python main.py --config experiments/exp_agnews_jina_v3_biencoder.yaml

# 3. Qwen (Largest)
!python main.py --config experiments/exp_agnews_qwen_embedding.yaml

# 4. BGE (Baseline - zaten var)
# Sonuçlar: results/raw/agnews_bge_description_metrics.json
```

**Beklenen sıralama:**
1. GTE veya Jina v3: 78-80% F1
2. BGE-m3: 77-79% F1
3. Qwen: 77-80% F1 (büyük ama belki overparameterized)

### Faz 2: Same-Family Hybrid (2 deney)

**Amaç:** Aynı aileden encoder+reranker sinerji yaratır mı?

```bash
# 1. BGE + BGE (zaten var)
# Sonuçlar: results/raw/agnews_bge_bge_reranker_metrics.json

# 2. Jina v3 + Jina v2 reranker
!python main.py --config experiments/exp_agnews_jina_v3_hybrid.yaml
```

**Hipotez:** Same-family daha iyi olmalı (+1-2%)

### Faz 3: Cross-Family Hybrid (2 deney)

**Amaç:** En iyi bi-encoder + en iyi reranker kombinasyonu?

```bash
# 1. GTE + BGE reranker
!python main.py --config experiments/exp_agnews_gte_bge_hybrid.yaml

# 2. Jina v3 + BGE reranker
!python main.py --config experiments/exp_agnews_jina_v3_bge_hybrid.yaml
```

**Hipotez:** Cross-family da iyi olabilir (diversity benefit)

## 📊 Beklenen Sonuçlar

### Bi-encoder Only:

| Model | Beklenen F1 | Neden |
|-------|-------------|-------|
| GTE | 78-80% | English'e optimize |
| Jina v3 | 78-80% | Task-adapted |
| BGE-m3 | 77-79% | Solid baseline |
| Qwen | 77-80% | Büyük ama belki overkill |

### Hybrid:

| Kombinasyon | Beklenen F1 | Artış |
|-------------|-------------|-------|
| BGE + BGE | 79-81% | +2% (baseline) |
| Jina v3 + Jina | 80-82% | +2-3% (optimized) |
| GTE + BGE | 80-82% | +2-3% (best of both) |
| Jina v3 + BGE | 80-82% | +2-3% (strong combo) |

## 🎯 Araştırma Soruları

### RQ1: Hangi bi-encoder en iyi?
**Cevap:** Faz 1 sonuçlarından

### RQ2: Hybrid pipeline ne kadar iyileştirir?
**Cevap:** Bi-encoder vs Hybrid karşılaştırması

### RQ3: Same-family vs cross-family?
**Cevap:** Faz 2 vs Faz 3 karşılaştırması

### RQ4: Label semantics etkisi?
**Cevap:** description vs multi_description (zaten test ettik)

## 🚀 Çalıştırma Komutları (Colab)

### Tek Seferde Hepsi:

```python
# Faz 1: Bi-encoders
configs_phase1 = [
    "experiments/exp_agnews_gte_biencoder.yaml",
    "experiments/exp_agnews_jina_v3_biencoder.yaml",
    "experiments/exp_agnews_qwen_embedding.yaml",
]

for config in configs_phase1:
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print('='*60)
    !python main.py --config {config}

# Faz 2 & 3: Hybrid
configs_phase2 = [
    "experiments/exp_agnews_jina_v3_hybrid.yaml",
    "experiments/exp_agnews_gte_bge_hybrid.yaml",
    "experiments/exp_agnews_jina_v3_bge_hybrid.yaml",
]

for config in configs_phase2:
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print('='*60)
    !python main.py --config {config}

# Sonuçları topla
!python compare_results.py
```

**Tahmini süre:** 
- Faz 1: ~10 dakika (4 bi-encoder)
- Faz 2+3: ~15 dakika (3 hybrid)
- **Toplam: ~25 dakika** (GPU'da)

## 📈 Sonuç Analizi

### Tablo 1: Bi-encoder Karşılaştırması

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| BGE-m3 | 77.6% | 77.1% | 77.1% |
| GTE | TBD | TBD | TBD |
| Jina v3 | TBD | TBD | TBD |
| Qwen | TBD | TBD | TBD |

### Tablo 2: Hybrid Karşılaştırması

| Encoder | Reranker | Macro F1 | Artış |
|---------|----------|----------|-------|
| BGE | BGE | 79.4% | +2.3% |
| Jina v3 | Jina | TBD | TBD |
| GTE | BGE | TBD | TBD |
| Jina v3 | BGE | TBD | TBD |

## 💡 Makale İçin Katkılar

Bu deneylerle şunları gösterebiliriz:

1. ✅ **Model karşılaştırması:** 4 SOTA bi-encoder sistematik test
2. ✅ **Hybrid analizi:** Same-family vs cross-family
3. ✅ **Label semantics:** description vs multi_description
4. ✅ **Robustness:** Farklı model aileleri arasında tutarlılık
5. ✅ **Practical insights:** Hangi kombinasyon en iyi?

## 🎓 Beklenen Bulgular

**Hipotezler:**

1. **GTE veya Jina v3 en iyi bi-encoder** olacak (English-optimized/task-adapted)
2. **Hybrid her zaman iyileştirecek** (~2-3%)
3. **Same-family약간 daha iyi** olabilir (ama fark küçük)
4. **Multi-description > description** (label semantics önemli)
5. **En iyi kombinasyon 81-83% F1** ulaşacak

## 📝 Sonraki Adımlar

Deneyler bittikten sonra:

1. ✅ `compare_results.py` çalıştır
2. ✅ `notebooks/02_error_analysis.ipynb` ile hata analizi
3. ✅ `notebooks/03_tables_and_plots.ipynb` ile tablo/grafik
4. ✅ En iyi modeli tam dataset'le test (max_samples: null)
5. ✅ Makale taslağı yaz

## 🚀 Hemen Başla!

```python
# Colab'da:
!python main.py --config experiments/exp_agnews_gte_biencoder.yaml
```

**Hazır! Başlayalım!** 🎯