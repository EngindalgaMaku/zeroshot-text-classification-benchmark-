# Geliştirilmiş Türkçe Offensive Detection Deneyleri

## Problem
Mevcut model "offensive" kavramını tam olarak anlamıyor. Sadece %9.9 recall ile çalışıyor, yani offensive içeriğin %90'ını kaçırıyor.

## Çözüm Önerisi
Model için daha detaylı, Türkçe'ye özgü label tanımları oluşturuldu. İki farklı versiyonu test edebiliriz:

### 1. Improved Labels (Geliştirilmiş Tanımlar)
**Dosya:** `experiments/exp_turkish_improved_labels.yaml`

- Offensive kavramının Türkçe açıklaması
- Türkçe küfür örnekleri (amk, aq, sik-, orospu, vb.)
- Hem offensive hem non-offensive için örnekler
- Orta düzeyde detaylı

### 2. Explicit Labels (Çok Detaylı Tanımlar)  
**Dosya:** `experiments/exp_turkish_explicit_labels.yaml`

- Çok detaylı offensive tanımı
- Kategorilere ayrılmış küfür listesi:
  - A) Türkçe Küfürler (amk, sik-, orospu, vb.)
  - B) Hakaretler (salak, aptal, mal, vb.)
  - C) Nefret Söylemi (etnik, dini, cinsel yönelimle ilgili)
  - D) Tehdit ve Şiddet İçerikli Dil
- "Tek bir küfür bile tüm metni offensive yapar" kuralı açık şekilde belirtilmiş

## Nasıl Çalıştırılır

### Opsiyon 1: Improved Labels
```bash
python main.py experiments/exp_turkish_improved_labels.yaml
```

### Opsiyon 2: Explicit Labels (Önerilen)
```bash
python main.py experiments/exp_turkish_explicit_labels.yaml
```

## Beklenen İyileşmeler

| Metrik | Mevcut | Hedef |
|--------|--------|-------|
| Recall | 9.9% | 60%+ |
| Precision | 55.4% | 50%+ |
| F1-Score | 16.9% | 55%+ |
| Accuracy | 51.0% | 70%+ |

## Neden Bu Çalışmalı?

1. **Label Descriptions are Critical**: Modeller özellikle zero-shot/few-shot senaryolarda label açıklamalarına çok güveniyor
2. **Turkish-Specific**: Mevcut tanımlar muhtemelen İngilizce-merkezli, Türkçe küfür kalıplarını bilmiyor
3. **Explicit Examples**: Model "amk", "aq", "sik-" gibi kelimelerin offensive olduğunu açıkça öğreniyor
4. **Rule-Based Guidance**: "Tek bir küfür bile offensive yapar" gibi kurallar modelin karar mekanizmasını güçlendiriyor

## Sonraki Adımlar

1. Her iki deney konfigürasyonunu çalıştır
2. Sonuçları karşılaştır (`analyze_turkish_results.py` kullanarak)
3. Hangisi daha iyi çalışıyorsa, o tanımları baz alarak fine-tune et
4. Gerekirse daha da detaylı Türkçe küfür listeleri ekle

## Alternatif İyileştirmeler

Eğer label descriptions yeterli olmazsa:
- Türkçe pre-trained model kullan (BERTurk, etc.)
- Explicit keyword matching feature'ları ekle
- Threshold'u düşür (0.5'ten 0.2'ye)
- Daha fazla few-shot örnek ekle (3'ten 10'a)