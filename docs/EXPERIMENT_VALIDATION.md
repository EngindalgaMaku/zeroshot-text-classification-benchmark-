# Experiment Configuration Validation

## ✅ Doğrulama Sonuçları

### 🎲 Random Seed Sabitleme
**EVET** - Reproducibility için seed=42 kullanılıyor

**Kod:** `main.py`
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Kullanım Yerleri:**
1. ✅ Dataset sampling: `dataset.shuffle(seed=42).select(range(max_samples))`
2. ✅ Global random state: `set_seed(42)` at startup
3. ✅ NumPy operations: `np.random.seed(42)`
4. ✅ PyTorch operations: `torch.manual_seed(42)`
5. ✅ CUDA operations: `torch.cuda.manual_seed_all(42)`

**Etki:**
- Aynı config ile her çalıştırmada aynı sonuçlar
- Dataset sampling her zaman aynı örnekleri seçer
- Model inference deterministik (CUDA hariç bazı operasyonlar)

### 1️⃣ Test Split Kullanımı
**EVET** - Tüm datasetler `test` split kullanıyor
- ✅ AG News: `split: test`
- ✅ DBpedia-14: `split: test`
- ✅ Yahoo Answers: `split: test`
- ✅ Banking77: `split: test`
- ✅ 20 Newsgroups: `split: test`
- ✅ GoEmotions: `split: test`
- ⚠️ Twitter Financial: `split: validation` (bu dataset'te test split yok)

### 2️⃣ Macro-F1 Metriği
**EVET** - Tüm datasetlerde aynı metric
```yaml
evaluation:
  metrics:
    - accuracy
    - macro_f1
    - per_class_f1
```
- Macro-F1: Her class için F1 hesapla, ortalamasını al (class imbalance'a duyarlı değil)
- Weighted-F1: Class sayısına göre ağırlıklı ortalama
- Accuracy: Doğru tahmin oranı

### 3️⃣ Label Description Formatı
**EVET** - Tüm datasetler `label_mode: description` kullanıyor

**Format:**
```python
"description": {
    0: ["This text is about international events, global politics..."],
    1: ["This text is about sports, matches, teams, athletes..."],
    # ...
}
```

**Örnekler:**
- AG News: "This text is about international events, global politics, diplomacy..."
- Banking77: "The user wants to activate their card or asking how to activate it."
- GoEmotions: "This text expresses admiration, respect, appreciation..."
- 20 Newsgroups: "This text discusses atheism, religious skepticism..."

**Tutarlılık:** ✅ Tüm descriptions aynı template'i takip ediyor:
- "This text is about..." (topic classification)
- "This text describes..." (entity classification)
- "This text expresses..." (emotion/sentiment)
- "The user wants to..." (intent classification)

### 4️⃣ Prompt / Template
**EVET** - Sabit template kullanılıyor

**Kod:** `src/pipeline.py`
```python
def predict_biencoder(texts, label_texts, label_ids, encoder, normalize=True, batch_size=32):
    # Encode texts
    text_embeddings = encoder.encode(texts, batch_size=batch_size)
    
    # Encode labels
    label_embeddings = encoder.encode(label_texts, batch_size=batch_size)
    
    # Compute cosine similarity
    similarities = cosine_similarity(text_embeddings, label_embeddings)
```

**INSTRUCTOR için özel handling:**
```python
# INSTRUCTOR uses task-specific instructions
if "instructor" in model_name.lower():
    instruction = "Represent the text for classification:"
    texts = [[instruction, text] for text in texts]
```

**Tutarlılık:** ✅ Tüm modeller için aynı pipeline, sadece INSTRUCTOR için instruction ekleniyor

### 5️⃣ Embedding Normalization
**EVET** - Tüm datasetlerde kullanılıyor
```yaml
pipeline:
  normalize_embeddings: true
```

**Kod:** `src/pipeline.py`
```python
if normalize:
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    label_embeddings = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)
```

**Neden önemli:**
- Cosine similarity için gerekli
- Embedding magnitude farklarını ortadan kaldırır
- Tüm modeller için adil karşılaştırma sağlar

---

## 📊 Sample Size Kontrolü

| Dataset | Split | Max Samples | Actual |
|---------|-------|-------------|--------|
| AG News | test | 1000 | 1000 |
| DBpedia-14 | test | 1000 | 1000 |
| Yahoo Answers | test | 1000 | 1000 |
| Banking77 | test | 1000 | 1000 |
| Twitter Financial | validation | 1000 | 1000 |
| 20 Newsgroups | test | 2000 | 2000 |
| GoEmotions | test | 1000 | 1000 |

**Not:** 20 Newsgroups 2000 sample kullanıyor (daha büyük dataset)

---

## 🔍 Potansiyel Sorunlar

### ⚠️ Twitter Financial - Validation Split
- Diğer datasetler `test` split kullanırken bu `validation` kullanıyor
- Sebep: Bu dataset'te test split yok
- Etki: Minimal - validation split de unseen data

### ⚠️ GoEmotions - Multi-label
- GoEmotions multi-label bir dataset (her text için birden fazla emotion)
- Çözüm: İlk emotion label'ı alınıyor (dominant emotion)
- Kod: `src/data.py` - `prepare_texts_and_labels()`
```python
if dataset_name == "go_emotions":
    converted_labels = []
    for label_list in labels:
        if isinstance(label_list, (list, tuple)) and len(label_list) > 0:
            converted_labels.append(label_list[0])  # Take first emotion
```

### ⚠️ 20 Newsgroups - Farklı Sample Size
- 2000 sample (diğerleri 1000)
- Sebep: Daha büyük ve daha zor dataset
- Etki: Daha güvenilir sonuçlar ama diğerleriyle tam karşılaştırılabilir değil

---

## ✅ Sonuç

**Tüm kritik parametreler tutarlı:**
0. ✅ Random seed sabitleme (seed=42)
1. ✅ Test split (Twitter hariç - validation)
2. ✅ Macro-F1 metriği
3. ✅ Description label formatı
4. ✅ Sabit pipeline/template
5. ✅ Embedding normalization

**Tablodaki sonuçlar güvenilir, karşılaştırılabilir ve reproducible!**

---

## 📝 Ek Notlar

### Model-Specific Settings
- **Qwen3**: `batch_size: 8` (büyük model, OOM önleme)
- **Diğerleri**: `batch_size: 32` (default)

### Label Mode Alternatifleri
- `name_only`: Sadece label adı ("sports", "business")
- `description`: Detaylı açıklama (kullanılan)
- `multi_description`: Birden fazla açıklama (kullanılmıyor)

### Metric Seçimi
- **Macro-F1**: Class imbalance'a duyarlı değil, her class eşit ağırlıkta
- **Weighted-F1**: Class sayısına göre ağırlıklı
- **Accuracy**: Basit doğruluk oranı

**Neden Macro-F1?**
- Banking77 gibi imbalanced datasetlerde daha adil
- Her class'ın performansını eşit değerlendirir
- Literatürde standart metric
