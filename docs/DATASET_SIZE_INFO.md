# Dataset Size Comparison

## Projedeki Datasetler ve Örnek Sayıları

### 📊 Dataset Özeti

| Dataset | Orijinal Boyut | Kullanılan Split | Bizim Kullandığımız | Oran |
|---------|---------------|------------------|---------------------|------|
| **AG News** | 7,600 (test) | test | 1,000 | 13.2% |
| **DBPedia-14** | 70,000 (test) | test | 1,000 | 1.4% |
| **Yahoo Answers** | 60,000 (test) | test | 1,000 | 1.7% |
| **Banking77** | 3,080 (test) | test | 1,000 | 32.5% |
| **20 Newsgroups** | 7,532 (test) | test | 2,000 | 26.6% |
| **Twitter Financial** | ~3,000 (test) | test | 1,000 | ~33% |

---

## 🔍 Detaylı Bilgi

### 1. AG News
- **Tam Adı:** AG's News Topic Classification Dataset
- **Sınıf Sayısı:** 4 (World, Sports, Business, Sci/Tech)
- **Orijinal Test Set:** 7,600 örnek
- **Kullanılan:** 1,000 örnek (13.2%)
- **Neden 1000?** Hızlı iterasyon için yeterli, istatistiksel olarak anlamlı

### 2. DBPedia-14
- **Tam Adı:** DBPedia Ontology Classification Dataset
- **Sınıf Sayısı:** 14 (Company, Artist, Athlete, vb.)
- **Orijinal Test Set:** 70,000 örnek
- **Kullanılan:** 1,000 örnek (1.4%)
- **Neden Az?** Çok büyük dataset, 1000 örnek bile yeterli temsil sağlıyor

### 3. Yahoo Answers Topics
- **Tam Adı:** Yahoo! Answers Topic Classification
- **Sınıf Sayısı:** 10 (Society, Science, Health, vb.)
- **Orijinal Test Set:** 60,000 örnek
- **Kullanılan:** 1,000 örnek (1.7%)
- **Neden Az?** Benzer şekilde büyük dataset

### 4. Banking77
- **Tam Adı:** Banking Domain Intent Classification
- **Sınıf Sayısı:** 77 (çok ince kategoriler)
- **Orijinal Test Set:** 3,080 örnek
- **Kullanılan:** 1,000 örnek (32.5%)
- **Not:** Nispeten küçük dataset, örneklerin büyük kısmını kullanıyoruz

### 5. 20 Newsgroups
- **Tam Adı:** 20 Newsgroups Text Classification
- **Sınıf Sayısı:** 20 (comp.graphics, sci.space, vb.)
- **Orijinal Test Set:** 7,532 örnek (SetFit version)
- **Kullanılan:** 2,000 örnek (26.6%)
- **Not:** En zorlu dataset, daha fazla örnek kullanılıyor

### 6. Twitter Financial News Sentiment
- **Tam Adı:** zeroshot/twitter-financial-news-sentiment
- **Sınıf Sayısı:** 3 (Bearish, Bullish, Neutral)
- **Orijinal Test Set:** ~3,000 örnek
- **Kullanılan:** 1,000 örnek (~33%)
- **Not:** Finansal sentiment, dengeli örnekleme

---

## 💡 Neden Bu Sayılar?

### Akademik Standartlar
- **1,000 örnek** zero-shot classification için standart
- Çoğu paper'da benzer sayılar kullanılır
- İstatistiksel olarak anlamlı sonuçlar için yeterli

### Pratik Nedenler
1. **Hız:** 1000 örnek = makul deneysel süre
2. **Maliyet:** API kullanımı için uygun
3. **GPU Memory:** Özellikle büyük modeller için kritik
4. **Reproducibility:** Sabit seed (42) ile tekrarlanabilir

### Sonuç Güvenilirliği
- 1000 örnek ile %95 güven aralığı: ±3.1%
- Macro F1 skorları güvenilir
- Model karşılaştırmaları için yeterli

---

## 📈 Örnek Sayısı Seçim Stratejisi

```python
# Genel kural:
if dataset_size > 10_000:
    max_samples = 1_000  # %1-10 arası
elif dataset_size > 5_000:
    max_samples = 2_000  # %20-40 arası
else:
    max_samples = min(1_000, dataset_size * 0.3)  # %30'a kadar
```

---

## 🎯 Öneriler

### Eğer Daha Fazla Örnek Kullanmak İsterseniz:

| Dataset | Önerilen Max | Avantaj | Dezavantaj |
|---------|--------------|---------|------------|
| AG News | 2,000-3,000 | Daha robust | +2-3x süre |
| DBPedia | 2,000-5,000 | 77 sınıf için iyi | +5-10x süre |
| Yahoo | 2,000-5,000 | Dengeli sonuçlar | +5-10x süre |
| Banking77 | Tamamı (3,080) | Full coverage | +3x süre |
| 20 News | 3,000-5,000 | 20 sınıf için ideal | +2-3x süre |

### Eğer GPU Memory Sorunu Varsa:

- AG News: 500 örnek bile yeterli
- DBPedia: 500-750
- Yahoo: 500-750
- Banking77: 500-750
- 20 News: 1,000 (minimum)

---

## ✅ Sonuç

**Mevcut seçim çok iyi!**
- Hızlı iterasyon ✅
- Güvenilir sonuçlar ✅
- Makul süre ✅
- Akademik standartlara uygun ✅

Eğer final paper için çalışıyorsanız, 20 Newsgroups ve Banking77'yi full kullanabilirsiniz, ama diğerleri için 1000 örnek tamamen yeterli.