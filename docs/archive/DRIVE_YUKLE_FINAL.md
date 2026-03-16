# 📤 Drive'a Yüklenecek Dosyalar - Final Liste

## 🚀 Hızlı Yöntem: Batch Script

### Windows:
```cmd
prepare_for_colab.bat
```

Bu otomatik olarak `zero_shot_colab` klasörünü oluşturur ve tüm gerekli dosyaları kopyalar.

## 📋 Drive'a Gidecek Dosyalar

### ✅ Ana Dosyalar
```
zero_shot_colab/
├── main.py                    # Ana çalıştırıcı
├── requirements.txt           # Bağımlılıklar
├── README.md                  # Dokümantasyon
├── GUNCEL_MODELLER_2026.md   # Model listesi (yeni!)
└── compare_results.py         # Sonuç karşılaştırma
```

### ✅ Kaynak Kod (src/)
```
src/
├── __init__.py
├── config.py                  # Config yükleme
├── data.py                    # Dataset işlemleri
├── labels.py                  # Label tanımları
├── encoders.py                # Bi-encoder (trust_remote_code=True)
├── rerankers.py               # Cross-encoder
├── pipeline.py                # Classification pipeline
├── llm_classifier.py          # LLM-based classification (yeni!)
├── metrics.py                 # Metrikler
├── runner.py                  # Deney runner (LLM support)
├── utils.py                   # Yardımcı fonksiyonlar
└── api_encoders.py            # API encoders (optional)
```

### ✅ Deney Config'leri (experiments/)

**NLP Encoders (5):**
```
experiments/
├── exp_agnews_baseline.yaml         # BGE-m3 (baseline)
├── exp_agnews_name_only.yaml        # BGE name-only
├── exp_agnews_deberta.yaml          # DeBERTa-v3 🆕
├── exp_agnews_roberta.yaml          # RoBERTa-large 🆕
├── exp_agnews_electra.yaml          # ELECTRA-large 🆕
└── exp_agnews_mpnet.yaml            # MPNet 🆕
```

**Hybrid:**
```
├── exp_agnews_bge_reranker.yaml     # BGE + BGE reranker
└── exp_agnews_mpnet_hybrid.yaml     # MPNet + BGE reranker
```

**LLM (2 - GÜNCEL 2026):**
```
├── exp_agnews_qwen_3b_llm.yaml      # Qwen3-8B 🆕
└── exp_agnews_llama_3_3.yaml        # Llama-3.3-8B 🆕
```

**Toplam:** 12 config

### ✅ Notebooks
```
notebooks/
├── 01_run_experiments.ipynb         # Ana notebook
├── 02_error_analysis.ipynb          # Hata analizi
└── 03_tables_and_plots.ipynb        # Görselleştirme
```

### ✅ Boş Klasörler (Otomatik Oluşur)
```
results/
├── raw/          # Ham sonuçlar (JSON, CSV)
├── tables/       # Tablolar (LaTeX, CSV)
└── plots/        # Grafikler (PNG)

data_cache/       # Dataset cache
```

## 🎯 Hangi Dosyalar ATLANACAK (Gereksiz)

### ❌ ATLA - Eski/Kullanılmayan:
```
❌ exp_agnews_gte_biencoder.yaml          # GTE çalışmıyor
❌ exp_agnews_jina_v3_*.yaml              # Jina v3 çalışmıyor
❌ exp_agnews_qwen_embedding.yaml         # Embedding versiyonu, LLM değil
❌ exp_agnews_multilingual_mpnet.yaml     # İsteğe bağlı
❌ exp_agnews_phi3_llm.yaml               # Eski, Llama kullanıyoruz
❌ exp_agnews_jina_api*.yaml              # API kullanmıyoruz
```

### ❌ ATLA - Dokümantasyon (İsteğe Bağlı):
```
❌ YENI_DENEY_PLANI.md                    # Eski plan
❌ GARANTILI_DENEY_PLANI.md               # Eski plan
❌ API_MODELLER_REHBER.md                 # API kullanmıyoruz
❌ NLP_SPECIFIC_MODELLER.md               # Taslak
❌ FINAL_MODEL_LISTESI.md                 # GUNCEL_MODELLER_2026.md var
```

## ✅ ÖZET: Ne Yapmalı?

### Adım 1: Batch Çalıştır
```cmd
prepare_for_colab.bat
```

### Adım 2: Kontrol Et
```cmd
dir zero_shot_colab
dir zero_shot_colab\src
dir zero_shot_colab\experiments
```

### Adım 3: Gereksizleri Temizle

**zero_shot_colab\experiments\ içinden SİL:**
```cmd
cd zero_shot_colab\experiments
del exp_agnews_gte*.yaml
del exp_agnews_jina_v3*.yaml
del exp_agnews_qwen_embedding.yaml
del exp_agnews_phi3*.yaml
del exp_agnews_jina_api*.yaml
del exp_agnews_multilingual*.yaml
```

### Adım 4: Drive'a Yükle

Sadece `zero_shot_colab` klasörünü:
```
MyDrive/zero_shot_colab/
```

## 📊 Final Config Listesi (12)

**Kullanılacak config'ler:**

1. ✅ `exp_agnews_baseline.yaml` - BGE-m3 baseline
2. ✅ `exp_agnews_name_only.yaml` - Label semantics
3. ✅ `exp_agnews_bge_reranker.yaml` - Hybrid (tested)
4. ✅ `exp_agnews_deberta.yaml` - NLP encoder
5. ✅ `exp_agnews_roberta.yaml` - NLP encoder
6. ✅ `exp_agnews_electra.yaml` - NLP encoder
7. ✅ `exp_agnews_mpnet.yaml` - General encoder
8. ✅ `exp_agnews_mpnet_hybrid.yaml` - Hybrid variant
9. ✅ `exp_agnews_qwen_3b_llm.yaml` - LLM (2026)
10. ✅ `exp_agnews_llama_3_3.yaml` - LLM (2026)

**Opsiyonel:**
11. `exp_agnews_comparison.yaml` - Template
12. `exp_agnews_hybrid.yaml` - Alternative hybrid

## 🚀 Colab'da İlk Çalıştırma

```python
# 1. Mount
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate
%cd /content/drive/MyDrive/zero_shot_colab

# 3. Install
!pip install -q -r requirements.txt

# 4. Test (hızlı)
!python main.py --config experiments/exp_agnews_baseline.yaml
```

## ✅ Kontrol Listesi

**Batch çalıştırmadan önce:**
- [x] Tüm src/ dosyaları güncel
- [x] LLM configs güncel (Qwen3, Llama-3.3)
- [x] runner.py LLM support var
- [x] llm_classifier.py mevcut

**Batch çalıştırdıktan sonra:**
- [ ] zero_shot_colab klasörü oluştu
- [ ] 12 config var (eski modeller silindi)
- [ ] src/ tam
- [ ] notebooks/ tam
- [ ] GUNCEL_MODELLER_2026.md var

**Drive'a yükledikten sonra:**
- [ ] Colab'da path doğru
- [ ] requirements.txt kurulu
- [ ] İlk deney çalıştı

## 🎯 Sonuç

**Ne yüklenecek:**
- Ana kod (main.py, src/)
- 12 config (güncel modeller)
- 3 notebook
- Dokümantasyon

**Ne yüklenmeyecek:**
- Eski/çalışmayan modeller
- Gereksiz dokümantasyon
- Test dosyaları

**Toplam boyut:** ~50-100 KB (kod), modeller Colab'da indirilecek

Hazır! 🚀