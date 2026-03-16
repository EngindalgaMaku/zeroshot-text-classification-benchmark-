# Sadece Qwen + 20 Newsgroups Deneyini Colab'da Çalıştırma

## 🚀 Hızlı Adımlar

### 1. Colab'ı Aç
https://colab.research.google.com/

### 2. Yeni Notebook Oluştur

### 3. Bu Kodu Çalıştır:

```python
# Adım 1: Kurulum
!pip install sentence-transformers datasets scikit-learn

# Adım 2: Kodları hazırla
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Adım 3: Veriyi yükle
print("Loading dataset...")
dataset = load_dataset("SetFit/20_newsgroups", split="test")
dataset = dataset.shuffle(seed=42).select(range(2000))

texts = dataset["text"]
true_labels = dataset["label"]

# Adım 4: Label descriptions
label_descriptions = [
    "This text discusses atheism, religious skepticism, secular humanism, or non-religious philosophy.",
    "This text discusses computer graphics, image processing, visualization, rendering, or graphical software.",
    "This text discusses Microsoft Windows operating system issues, tips, or questions.",
    "This text discusses IBM PC hardware, components, upgrades, or technical specifications.",
    "This text discusses Apple Macintosh hardware, components, or technical specifications.",
    "This text discusses X Window System, Unix graphical interface, or related software.",
    "This text is a for-sale advertisement, marketplace listing, or commercial offer.",
    "This text discusses automobiles, cars, driving, automotive technology, or vehicle maintenance.",
    "This text discusses motorcycles, bikes, riding, or motorcycle maintenance.",
    "This text discusses baseball, MLB, baseball players, games, or statistics.",
    "This text discusses hockey, NHL, hockey players, games, or ice hockey.",
    "This text discusses cryptography, encryption, security algorithms, or cryptographic systems.",
    "This text discusses electronics, circuits, electronic components, or electrical engineering.",
    "This text discusses medicine, medical conditions, healthcare, or medical research.",
    "This text discusses space, astronomy, space exploration, NASA, or astrophysics.",
    "This text discusses Christianity, Christian faith, Bible, or Christian theology.",
    "This text discusses gun politics, firearms, gun rights, or gun control debates.",
    "This text discusses Middle East politics, conflicts, or geopolitical issues in that region.",
    "This text discusses general political topics, political debates, or miscellaneous political issues.",
    "This text discusses general religious topics, interfaith dialogue, or miscellaneous religious matters."
]

# Adım 5: Model yükle
print("Loading Qwen model...")
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

# Adım 6: Embeddings
print("Encoding texts...")
text_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
label_embeddings = model.encode(label_descriptions, show_progress_bar=True)

# Adım 7: Similarity ve prediction
print("Computing predictions...")
similarities = np.dot(text_embeddings, label_embeddings.T)
predictions = np.argmax(similarities, axis=1)
confidences = np.max(similarities, axis=1) / np.linalg.norm(label_embeddings, axis=1)

# Adım 8: Metrics
accuracy = accuracy_score(true_labels, predictions)
macro_f1 = f1_score(true_labels, predictions, average='macro')
weighted_f1 = f1_score(true_labels, predictions, average='weighted')
report = classification_report(true_labels, predictions, output_dict=True)

# Adım 9: Sonuçları göster
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Mean Confidence: {confidences.mean():.4f}")

# Adım 10: JSON kaydet
results = {
    "experiment_name": "20newsgroups_qwen3",
    "dataset_name": "SetFit/20_newsgroups",
    "num_samples": len(texts),
    "num_classes": 20,
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "macro_precision": float(report['macro avg']['precision']),
    "macro_recall": float(report['macro avg']['recall']),
    "mean_confidence": float(confidences.mean()),
    "classification_report": {str(k): v for k, v in report.items()}
}

# JSON'u göster
print("\n" + "="*50)
print("JSON OUTPUT (Bu kısmı kopyalayın!)")
print("="*50)
print(json.dumps(results, indent=2))

# İsteğe bağlı: Drive'a kaydet
from google.colab import drive
drive.mount('/content/drive')

with open('/content/drive/MyDrive/20newsgroups_qwen3_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to Google Drive!")
```

---

## 📥 Sonucu Lokal'e Getirme

### Yöntem 1: JSON'u Kopyala
1. Colab'daki "JSON OUTPUT" kısmını kopyala
2. Lokal'de `results/raw/20newsgroups_qwen3_metrics.json` dosyası oluştur
3. Yapıştır ve kaydet

### Yöntem 2: Drive'dan İndir
1. Google Drive'ınızı aç
2. `20newsgroups_qwen3_metrics.json` dosyasını bul
3. İndir ve `results/raw/` klasörüne koy

---

## 🎨 Visualizations'ı Güncelle

```bash
python generate_beautiful_plots.py
```

✅ Hepsi bu kadar!