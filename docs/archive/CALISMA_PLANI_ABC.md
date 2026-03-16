# 🎯 A-B-C Çalışma Planı

## ✅ MLM Test Başarılı!
- RoBERTa-base: 60% accuracy (10 sample)
- MLM yaklaşımı çalışıyor!

---

## 📋 FAZ A: MLM'i Tamamla (1-2 Gün)

### A1: Runner Entegrasyonu ✅ ŞİMDİ
- [ ] `src/runner.py` güncelle (MLM desteği ekle)
- [ ] Config'den MLM parametreleri oku
- [ ] MLM prediction pipeline'ı entegre et

### A2: 1000 Örnek Test
- [ ] AG News 1000 sample
- [ ] Hedef: 65-75% accuracy
- [ ] Sonuçları kaydet

### A3: DeBERTa Test (Opsiyonel)
- [ ] Protobuf kur
- [ ] DeBERTa test et
- [ ] RoBERTa ile karşılaştır

**Çıktı:** 
- `results/mlm/agnews_roberta_mlm_metrics.json`
- `results/mlm/agnews_roberta_mlm_predictions.csv`

---

## 📋 FAZ B: Dataset Ekle (2-3 Gün)

### B1: DBpedia
- [ ] DBpedia yükle
- [ ] Label words tanımla
- [ ] MLM test et

### B2: SST-2 (Sentiment)
- [ ] SST-2 yükle
- [ ] Sentiment label words
- [ ] MLM test et

### B3: TREC (Question)
- [ ] TREC yükle
- [ ] Question label words
- [ ] MLM test et

**Çıktı:**
- 3 dataset × MLM = 3 deney
- Her biri için metrics + predictions

---

## 📋 FAZ C: LLM Ekle (3-4 Gün)

### C1: Çalışan LLM Bul
- [ ] Mistral-7B test et
- [ ] Zephyr test et
- [ ] Ya da daha küçük model

### C2: LLM Prompting
- [ ] Prompt template tasarla
- [ ] Generation-based classification
- [ ] Parse response

### C3: Tüm Datasetlerde Test
- [ ] AG News
- [ ] DBpedia
- [ ] SST-2
- [ ] TREC

**Çıktı:**
- 4 dataset × LLM = 4 deney
- LLM vs MLM vs Embedding karşılaştırması

---

## 🎯 TOPLAM HEDEF

**3 Approach × 4 Dataset = 12 Deney Grubu**

| Dataset | Embedding | MLM | LLM |
|---------|-----------|-----|-----|
| AG News | ✅ 77% | 🔄 | ❌ |
| DBpedia | ❌ | ❌ | ❌ |
| SST-2 | ❌ | ❌ | ❌ |
| TREC | ❌ | ❌ | ❌ |

---

## 📅 Timeline

- **Faz A:** 1-2 gün
- **Faz B:** 2-3 gün  
- **Faz C:** 3-4 gün
- **Analiz:** 2-3 gün
- **Makale:** 1 hafta

**TOPLAM:** ~3 hafta

---

## 🚀 ŞİMDİ: Faz A1 - Runner Entegrasyonu

**Yapılacaklar:**
1. `src/runner.py` güncelle
2. MLM pipeline ekle
3. Config'ten parametreleri oku
4. Test et

**Başlıyoruz!** 🎯