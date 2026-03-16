# Scripts Guide

Bu klasördeki scriptlerin ne yaptığını ve nasıl çalıştırılacağını açıklar.
Tüm komutlar proje kök dizininden çalıştırılmalıdır.

---

## Kullanım Senaryoları

### Senaryo A — Sadece Deneyleri Çalıştır

Tüm 63 deneyi (9 dataset × 7 model) çalıştırmak istiyorsanız:

```cmd
cmd /c "conda activate zeroshot && python scripts/fix_and_run_all_experiments.py"
```

Veya conda ortamı zaten aktifse:

```bash
python scripts/fix_and_run_all_experiments.py
```

**Ne yapar:** `experiments/` altındaki tüm config dosyalarını sırayla çalıştırır.
Sonuçlar `results/raw/` altına `*_metrics.json` ve `*_predictions.csv` olarak kaydedilir.

**Süre:** GPU'ya bağlı olarak 6-10 saat.

> Sadece label formülasyon deneylerini çalıştırmak için:
> ```bash
> python scripts/run_label_formulation_experiments.py
> ```

---

### Senaryo B — Sadece Raporlama / Analiz Yap

Deneyler zaten tamamlandıysa, sadece figür ve tabloları üretmek için:

```cmd
cmd /c "conda activate zeroshot && python scripts/regenerate_all.py"
```

Veya conda ortamı zaten aktifse:

```bash
python scripts/regenerate_all.py
```

**Ne yapar:** `results/raw/` altındaki mevcut sonuçları okuyup tüm figür, tablo ve
istatistiksel analizleri yeniden üretir.

**Süre:** ~5-15 dakika (GPU gerektirmez).

> Yavaş adımları (confusion matrices, task characteristics) atlamak için:
> ```bash
> python scripts/regenerate_all.py --skip-slow
> ```

---

### Senaryo C — Sıfırdan Tam Çalıştırma

Hem deneyleri hem raporlamayı sıfırdan yapmak için:

```cmd
cmd /c "conda activate zeroshot && python scripts/fix_and_run_all_experiments.py && python scripts/regenerate_all.py"
```

---

### Senaryo D — Tek Bir Deney

Belirli bir model-dataset kombinasyonunu test etmek için:

```bash
python main.py --config experiments/exp_ag_news_bge.yaml
```

---

## Hızlı Başlangıç

Tüm analizleri sıfırdan üretmek için tek komut:

```bash
python scripts/regenerate_all.py
```

---
## Script Referansı

### 1. regenerate_all.py — Ana Orkestratör

Tüm analiz scriptlerini doğru sırayla çalıştırır.

```bash
# Tüm adımları çalıştır
python scripts/regenerate_all.py

# Yavaş adımları atla (task characteristics, confusion matrices)
python scripts/regenerate_all.py --skip-slow
```

**Çıktılar:** `results/tables/`, `results/plots/`, `results/statistical_analysis/`,
`results/stability_analysis/`, `results/task_characteristics/`

---

### 2. collect_all_results.py — Sonuç Toplayıcı

`results/raw/` altındaki tüm JSON dosyalarını okuyup tek bir CSV'ye toplar.
Diğer tüm analiz scriptleri bu CSV'ye bağımlıdır — **önce bu çalışmalıdır.**

```bash
python scripts/collect_all_results.py
```

**Çıktı:** `results/MULTI_DATASET_RESULTS.csv`

---

### 3. statistical_analysis.py — İstatistiksel Testler

Friedman testi + Nemenyi post-hoc + güç analizi yapar.

```bash
python scripts/statistical_analysis.py
```

**Çıktılar:**
- `results/statistical_analysis/statistical_tests.json` — ham test sonuçları
- `results/statistical_analysis/POWER_ANALYSIS.md` — güç analizi raporu
- `results/statistical_analysis/README.md` — özet

**Bağımlılık:** `collect_all_results.py` önce çalışmış olmalı.

---

### 4. generate_publication_heatmap.py — Model × Dataset Heatmap

7 model × 9 dataset Macro-F1 ısı haritasını yayın kalitesinde üretir.

```bash
python scripts/generate_publication_heatmap.py
```

**Çıktılar:** `results/plots/heatmap_publication.pdf`, `.eps`, `.png`

---

### 5. generate_critical_difference_diagram.py — Kritik Fark Diyagramı

Nemenyi testine dayalı kritik fark (CD) diyagramını üretir.
Hangi modeller arasında istatistiksel olarak anlamlı fark olduğunu gösterir.

```bash
python scripts/generate_critical_difference_diagram.py
```

**Çıktılar:** `results/plots/critical_difference_diagram.pdf`, `.eps`, `.png`

**Bağımlılık:** `statistical_analysis.py` önce çalışmış olmalı.

---

### 6. generate_label_formulation_figure.py — Label Formülasyon Karşılaştırması

`name_only` vs `description` label modlarının performans farkını görselleştirir.

```bash
python scripts/generate_label_formulation_figure.py
```

**Çıktılar:** `results/plots/label_formulation_comparison.pdf`, `.eps`, `.png`

**Bağımlılık:** `experiments/label_formulation/` altındaki deneylerin tamamlanmış olması gerekir.

---

### 7. generate_task_type_analysis.py — Task Tipi Analizi

Modellerin farklı task tiplerine (topic, intent, sentiment, emotion) göre performansını
gruplu bar chart olarak gösterir.

```bash
python scripts/generate_task_type_analysis.py
```

**Çıktılar:** `results/plots/task_type_analysis.pdf`, `.eps`, `.png`

---

### 8. analyze_model_stability.py — Stabilite Metrikleri

Her modelin dataset'ler arası varyasyon katsayısını (CV) hesaplar.
Stabilite sıralaması tablosu üretir.

```bash
python scripts/analyze_model_stability.py
```

**Çıktılar:** `results/stability_analysis/` altında CSV ve JSON dosyaları

---

### 9. visualize_model_stability.py — Stabilite Görselleştirme

Ortalama performans vs stabilite scatter plot'u üretir.
Quadrant çizgileri ve model etiketleri içerir.

```bash
python scripts/visualize_model_stability.py
```

**Çıktılar:**
- `results/plots/model_stability_scatter.pdf`, `.eps`, `.png`
- `results/plots/model_stability_ranking.pdf`, `.eps`, `.png`
- `results/plots/performance_stability_comparison.pdf`, `.eps`, `.png`

**Bağımlılık:** `analyze_model_stability.py` önce çalışmış olmalı.

---

### 10. analyze_task_characteristics.py — Task Karakteristikleri

Her dataset için sınıf sayısı, ortalama metin uzunluğu ve label semantik benzerliğini hesaplar.
Bunları model performansıyla korelasyon analizi yapar.

```bash
python scripts/analyze_task_characteristics.py
```

**Çıktılar:** `results/task_characteristics/` altında CSV ve JSON dosyaları

**Not:** Sentence embedding hesapladığı için yavaş çalışabilir (~2-5 dk).

---

### 11. visualize_task_characteristics.py — Task Karakteristik Grafikleri

Sınıf sayısı, metin uzunluğu ve label benzerliği vs Macro-F1 scatter plot'larını üretir.

```bash
python scripts/visualize_task_characteristics.py
```

**Çıktılar:**
- `results/plots/task_char_num_classes.pdf`, `.eps`, `.png`
- `results/plots/task_char_text_length.pdf`, `.eps`, `.png`
- `results/plots/task_char_label_similarity.pdf`, `.eps`, `.png`
- `results/plots/task_characteristics_combined.pdf`, `.eps`, `.png`

**Bağımlılık:** `analyze_task_characteristics.py` önce çalışmış olmalı.

---

### 12. analyze_error_patterns.py — Hata Örüntüsü Analizi

Her dataset için en çok karıştırılan sınıf çiftlerini tespit eder.
GoEmotions ve Yahoo Answers için özel hata analizi yapar.

```bash
python scripts/analyze_error_patterns.py
```

**Çıktılar:** `results/tables/error_patterns_detailed.csv`

**Bağımlılık:** `results/raw/` altında `*_predictions.csv` dosyaları mevcut olmalı.

---

### 13. generate_confusion_matrices.py — Confusion Matrix Heatmap'leri

Temsili dataset'ler (AG News, Banking77, GoEmotions) için confusion matrix görsellerini üretir.

```bash
python scripts/generate_confusion_matrices.py
```

**Çıktılar:** `results/plots/confusion_matrices/` altında PDF/PNG dosyaları

**Bağımlılık:** `results/raw/` altında `*_predictions.csv` dosyaları mevcut olmalı.
**Not:** Yavaş çalışabilir — `regenerate_all.py --skip-slow` ile atlanabilir.

---

### 14. fix_and_run_all_experiments.py — Toplu Deney Çalıştırıcı

Tüm 49 deneyi sırayla çalıştırır. Hata durumunda devam eder ve log tutar.

```bash
python scripts/fix_and_run_all_experiments.py
```

**Çıktılar:** `results/raw/` altında `*_metrics.json` ve `*_predictions.csv` dosyaları

**Uyarı:** GPU'ya bağlı olarak 6-10 saat sürebilir.

---

### 15. run_label_formulation_experiments.py — Label Formülasyon Deneyleri

`name_only` ve `description` modları için karşılaştırma deneylerini çalıştırır.

```bash
python scripts/run_label_formulation_experiments.py
```

**Çıktılar:** `results/raw/` altında label formülasyon sonuçları

---

### 16. archive_old_results.py — Eski Sonuçları Arşivle

`results/raw/` altındaki mevcut sonuçları `results/archive/` altına taşır.
Yeni deneyler çalıştırmadan önce kullanılır.

```bash
python scripts/archive_old_results.py
```

---

### 17. verify_seed_implementation.py — Seed Doğrulama

`main.py` içindeki seed implementasyonunu doğrular.
Seed'in tüm kütüphanelere (Python, NumPy, PyTorch, CUDA) uygulandığını kontrol eder.

```bash
python scripts/verify_seed_implementation.py
```

---

## Bağımlılık Sırası

```
fix_and_run_all_experiments.py   ← deneyleri çalıştır
        ↓
collect_all_results.py           ← JSON'ları CSV'ye topla
        ↓
statistical_analysis.py          ← Friedman + Nemenyi
        ↓
generate_publication_heatmap.py
generate_critical_difference_diagram.py
generate_label_formulation_figure.py
generate_task_type_analysis.py
        ↓
analyze_model_stability.py  →  visualize_model_stability.py
analyze_task_characteristics.py  →  visualize_task_characteristics.py
analyze_error_patterns.py  →  generate_confusion_matrices.py
```

Veya hepsini tek seferde: `python scripts/regenerate_all.py`

---

## Archive Klasörü

`scripts/archive/` klasöründe geliştirme sürecinde kullanılan yardımcı scriptler bulunur
(config oluşturma, tek seferlik fix scriptleri, eski görselleştirme denemeleri vb.).
Bunlar artık aktif kullanımda değildir ama referans için saklanmıştır.


