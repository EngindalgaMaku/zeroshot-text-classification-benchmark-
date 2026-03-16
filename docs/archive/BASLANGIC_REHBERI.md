# 🚀 Başlangıç Rehberi

Proje tamamen hazır! İşte adım adım çalıştırma rehberi:

## ✅ Durum: %100 Hazır!

Tüm dosyalar oluşturuldu ve çalışmaya hazır. API key'e ihtiyacınız yok - tüm modeller HuggingFace'ten ücretsiz indirilecek.

## 🎯 İlk Adımlar (5 Dakika)

### Adım 1: Bağımlılıkları Kur

Projenin bulunduğu dizinde terminali açın ve şunu çalıştırın:

```bash
pip install -r requirements.txt
```

Bu işlem ~5-10 dakika sürebilir (modellere bağlı olarak).

### Adım 2: Kurulumu Test Et

```bash
python test_setup.py
```

Bu script şunları kontrol eder:
- ✅ Tüm paketler yüklü mü?
- ✅ Klasör yapısı doğru mu?
- ✅ GPU var mı? (opsiyonel)
- ✅ Modüller import edilebiliyor mu?

### Adım 3: İlk Deneyi Çalıştır

```bash
python main.py --config experiments/exp_agnews_baseline.yaml
```

**Bu deney:**
- AG News veri setinden 1000 örnek kullanır
- BAAI/bge-m3 modelini indirir (ilk seferde ~1GB)
- ~2-3 dakikada sonuç verir (GPU ile)
- Accuracy ~85-90% beklenir

## 📊 Sonuçları Görüntüleme

Deney bitince:

```bash
# Sonuç dosyalarını listele
dir results\raw

# Metrikleri görüntüle (Python ile)
python -c "import json; print(json.dumps(json.load(open('results/raw/agnews_bge_description_metrics.json')), indent=2))"
```

Ya da Python script içinde:

```python
import json
import pandas as pd

# Metrikleri oku
with open("results/raw/agnews_bge_description_metrics.json") as f:
    metrics = json.load(f)
    
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Tahminleri oku
df = pd.read_csv("results/raw/agnews_bge_description_predictions.csv")
print(f"\nDoğru tahmin: {df['correct'].sum()}/{len(df)}")
```

## 🔥 Farklı Deneyleri Deneyin

### 1. Minimal Label Modu (Daha Hızlı)

```bash
python main.py --config experiments/exp_agnews_name_only.yaml
```

Sadece sınıf isimleri kullanır: "world", "sports", "business", "technology"

### 2. Hybrid Pipeline (Daha İyi Sonuç)

```bash
python main.py --config experiments/exp_agnews_hybrid.yaml
```

Bi-encoder + Reranker kullanır. Genelde %2-5 daha iyi F1 verir.

### 3. Farklı Model Deneyin

`experiments/exp_comparison.yaml` dosyasını açın ve model ismini değiştirin:

```yaml
models:
  biencoder:
    name: jinaai/jina-embeddings-v3  # Bu satırı değiştirin
```

Popüler modeller:
- `BAAI/bge-m3` (çok dilli, hızlı)
- `jinaai/jina-embeddings-v3` (güçlü)
- `sentence-transformers/all-mpnet-base-v2` (hafif)

## 📈 Analiz Notebooks (Opsiyonel)

Google Colab veya Jupyter ile:

1. `notebooks/01_run_experiments.ipynb` → Deneyleri çalıştır
2. `notebooks/02_error_analysis.ipynb` → Hataları analiz et
3. `notebooks/03_tables_and_plots.ipynb` → Makale için tablolar

### Colab Kullanımı

1. Projeyi Google Drive'a yükle → `MyDrive/zero_shot_reliable_cls`
2. Colab'da `01_run_experiments.ipynb` aç
3. Drive'ı mount et
4. Hücreleri sırayla çalıştır

## ⚙️ Ayarlar ve Özelleştirme

### Veri Seti Boyutu Değiştirme

Config dosyasında:

```yaml
dataset:
  max_samples: 500  # Daha hızlı test için
  # max_samples: null  # Tüm veri seti için
```

### GPU vs CPU

Kod otomatik algılar. GPU kullanmak için:

```python
# GPU varsa otomatik kullanılır
# CPU'ya zorlamak için:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Batch Size (Bellek İçin)

`src/encoders.py` ve `src/pipeline.py` içinde:

```python
batch_size=32  # GPU ile
batch_size=8   # Bellek sorunu varsa
```

## 🎯 Tam Çalışma Akışı

### Senaryo: AG News Deney Serisi

```bash
# 1. Name only (baseline)
python main.py --config experiments/exp_agnews_name_only.yaml

# 2. Description
python main.py --config experiments/exp_agnews_baseline.yaml

# 3. Multi-description
python main.py --config experiments/exp_agnews_hybrid.yaml
```

### Sonuçları Karşılaştırın

```python
import json
import pandas as pd
from pathlib import Path

results = []
for f in Path("results/raw").glob("*_metrics.json"):
    with open(f) as fp:
        m = json.load(fp)
    results.append({
        "deney": m["experiment_name"],
        "accuracy": f"{m['accuracy']:.4f}",
        "macro_f1": f"{m['macro_f1']:.4f}",
    })

df = pd.DataFrame(results)
print(df.sort_values("macro_f1", ascending=False))
```

## 🔧 Sorun mu Yaşıyorsunuz?

### "Out of Memory" Hatası

```yaml
# Config dosyasında:
dataset:
  max_samples: 200  # Daha az örnek
```

### "Model indirilemiyor"

- İnternet bağlantınızı kontrol edin
- Firewall/proxy ayarlarını kontrol edin
- Manuel indirme: `git clone https://huggingface.co/BAAI/bge-m3 models/bge-m3`

### "Çok yavaş"

- Google Colab kullanın (GPU ile)
- Veya veri seti boyutunu azaltın

### Detaylı Sorun Giderme

`TROUBLESHOOTING.md` dosyasına bakın.

## 📝 Kendi Verilerinizi Kullanma

### 1. CSV Hazırlayın

```csv
text,label
"İlk metin buraya...",0
"İkinci metin...",1
```

### 2. Label Tanımları Ekleyin

`src/labels.py` dosyasına:

```python
LABEL_SETS["benim_verim"] = {
    "description": {
        0: ["İlk sınıf açıklaması"],
        1: ["İkinci sınıf açıklaması"],
    }
}
```

### 3. Config Oluşturun

`experiments/exp_benim_verim.yaml`:

```yaml
experiment_name: benim_deney

dataset:
  name: benim_verim  # Eğer özel yükleme gerekiyorsa src/data.py'yi güncelle
  split: test
  text_column: text
  label_column: label
  max_samples: 1000

task:
  label_mode: description

models:
  biencoder:
    name: BAAI/bge-m3
  reranker: null

pipeline:
  mode: biencoder
  normalize_embeddings: true

output:
  save_predictions: true
  save_metrics: true
  output_dir: results/raw
```

### 4. Çalıştırın

```bash
python main.py --config experiments/exp_benim_verim.yaml
```

## 💡 İpuçları

1. **İlk deney kısa olsun**: `max_samples: 100` ile başlayın
2. **GPU kullanın**: Colab ücretsiz GPU verir
3. **Sonuçları kaydedin**: Her deney results/raw/ altına otomatik kaydedilir
4. **Label'ları iyi yazın**: Detaylı açıklamalar genelde daha iyi sonuç verir
5. **Hybrid pipeline deneyin**: %2-5 daha iyi F1 beklenebilir

## 🎓 Makale Yazımı İçin

```bash
# Tüm deneyleri çalıştır
python main.py --config experiments/exp_agnews_name_only.yaml
python main.py --config experiments/exp_agnews_baseline.yaml
python main.py --config experiments/exp_agnews_hybrid.yaml

# Jupyter/Colab'da aç:
notebooks/03_tables_and_plots.ipynb

# LaTeX tabloları otomatik üretilir:
# results/tables/main_results.tex
# results/tables/top5_results.tex
```

## ❓ API Key Konusu

**Hiçbir API key gerekmez!** 

- ✅ HuggingFace modelleri **ücretsiz** ve **açık kaynak**
- ✅ Modeller **otomatik indirilir**
- ✅ İlk kullanımda önbelleğe alınır
- ✅ Sonraki kullanımlarda lokal cache'ten yüklenir

Eğer ileride OpenAI gibi ücretli API kullanmak isterseniz:
- `.env` dosyası oluşturun
- `OPENAI_API_KEY=...` ekleyin
- Ancak şu anki projede **buna gerek yok**!

## 🚀 Hemen Başlayın!

```bash
# 1. Kurulum
pip install -r requirements.txt

# 2. Test
python test_setup.py

# 3. İlk deney
python main.py --config experiments/exp_agnews_baseline.yaml

# 4. Sonuçları gör
python -c "import json; m=json.load(open('results/raw/agnews_bge_description_metrics.json')); print(f\"Accuracy: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}\")"
```

Başarılar! 🎉

---

**Sorularınız için:**
- README.md → Genel bilgi
- QUICKSTART.md → İngilizce hızlı başlangıç
- TROUBLESHOOTING.md → Sorun giderme
- Bu dosya → Türkçe adım adım rehber