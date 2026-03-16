# 📤 Drive'a Hangi Dosyaları Yüklemeli?

## ⚠️ 44000 Dosya = YANLIŞ!

**Sorun:** Python cache, git history, venv gibi gereksiz dosyalar var.

## ✅ SADECE Bunları Yükleyin:

### Gerekli Dosyalar (~50-100 dosya):

```
zero_shot_reliable_cls/
├── src/                    ✅ GEREKLI
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── labels.py
│   ├── encoders.py
│   ├── rerankers.py
│   ├── pipeline.py
│   ├── metrics.py
│   ├── runner.py
│   └── utils.py
├── experiments/            ✅ GEREKLI
│   ├── *.yaml (tüm config dosyaları)
├── notebooks/              ✅ GEREKLI
│   ├── *.ipynb (notebook'lar)
├── main.py                 ✅ GEREKLI
├── requirements.txt        ✅ GEREKLI
├── README.md              ✅ İSTEĞE BAĞLI
├── *.md (diğer docs)      ✅ İSTEĞE BAĞLI
├── check_gpu.py           ✅ İSTEĞE BAĞLI
├── test_setup.py          ✅ İSTEĞE BAĞLI
└── compare_results.py     ✅ İSTEĞE BAĞLI
```

### ❌ ATLAYIN (Yüklemeyin!):

```
❌ __pycache__/           Python cache
❌ .git/                  Git history (binlerce dosya!)
❌ venv/                  Virtual environment
❌ env/                   Virtual environment
❌ .venv/                 Virtual environment
❌ data_cache/            Dataset cache
❌ models/                Model cache
❌ results/               Eski sonuçlar (gerekirse sadece raw/)
❌ .vscode/               Editor ayarları
❌ .idea/                 Editor ayarları
❌ *.pyc                  Python compiled
❌ .DS_Store              Mac dosyası
❌ Thumbs.db              Windows dosyası
```

## 🚀 Hızlı Çözüm: Temiz Klasör Oluştur

### Yöntem 1: Gerekli Dosyaları Kopyala

Yeni bir klasör oluşturun:

```
zero_shot_colab/        ← Yeni temiz klasör
├── src/                (kopyala)
├── experiments/        (kopyala)
├── notebooks/          (kopyala)
├── main.py
├── requirements.txt
└── README.md
```

**Sonuç:** ~50-100 dosya, çok hızlı yüklenir!

### Yöntem 2: Seçici Yükle

1. **Önce src/ klasörünü** yükleyin
2. **Sonra experiments/** yükleyin
3. **Sonra notebooks/** yükleyin
4. **main.py** yükleyin
5. **requirements.txt** yükleyin

Bitti! ✅

## 📋 Komut Satırıyla (İleri Seviye)

Terminal'de projenizin klasöründe:

```bash
# Temiz klasör oluştur
mkdir ../zero_shot_colab

# Gerekli dosyaları kopyala
cp -r src ../zero_shot_colab/
cp -r experiments ../zero_shot_colab/
cp -r notebooks ../zero_shot_colab/
cp main.py ../zero_shot_colab/
cp requirements.txt ../zero_shot_colab/
cp README.md ../zero_shot_colab/
cp *.md ../zero_shot_colab/  # İsterseniz
```

Sonra `zero_shot_colab` klasörünü Drive'a yükleyin.

## 🔍 Hangi Dosyalar Gereksiz?

### .git klasörü (En Büyük):
- Git version history
- Binlerce dosya
- **GEREKSIZ** Colab için

### __pycache__:
- Python bytecode cache
- Colab'da yeniden oluşur
- **GEREKSIZ**

### venv/env/:
- Virtual environment
- 1000+ paket dosyası
- Colab kendi kurar
- **GEREKSIZ**

### data_cache/:
- HuggingFace dataset cache
- Colab'da yeniden indirilir
- **GEREKSIZ**

## ✅ Hangi Dosyalar Gerekli?

### src/ (Python kodları):
- ~10 dosya
- **ZORUNLU** - asıl mantık burada

### experiments/ (Config'ler):
- ~5-10 YAML dosyası
- **ZORUNLU** - deney ayarları

### notebooks/ (Colab interface):
- ~3 notebook
- **ZORUNLU** - Colab için

### main.py:
- 1 dosya
- **ZORUNLU** - giriş noktası

### requirements.txt:
- 1 dosya
- **ZORUNLU** - paket listesi

## 📊 Boyut Karşılaştırması:

| İçerik | Dosya Sayısı | Boyut |
|--------|--------------|-------|
| **TÜM proje (YANLIŞ)** | 44000+ | 2-5 GB ❌ |
| **Temiz proje (DOĞRU)** | 50-100 | 1-5 MB ✅ |

## 🎯 Önerilen Yöntem:

### Adım 1: Temiz Klasör

Masaüstünde yeni klasör:
```
Desktop/zero_shot_colab/
```

### Adım 2: Sadece Bu Klasörleri Kopyala

```
✅ src/
✅ experiments/
✅ notebooks/
```

### Adım 3: Bu Dosyaları Kopyala

```
✅ main.py
✅ requirements.txt
✅ README.md (opsiyonel)
```

### Adım 4: Drive'a Yükle

```
Desktop/zero_shot_colab/ → Sürükle → MyDrive
```

**Süre:** ~1-2 dakika ✨

## 🚫 Kesinlikle Yüklemeyin:

```
❌ .git/
❌ __pycache__/
❌ venv/
❌ env/
❌ .venv/
❌ node_modules/ (eğer varsa)
❌ .pytest_cache/
❌ .mypy_cache/
❌ data_cache/
❌ models/
❌ *.pyc
❌ .DS_Store
```

## 💡 Hızlı Kontrol:

Yüklemeden önce:

```bash
# Windows:
dir /s zero_shot_colab | find /c /v ""

# Mac/Linux:
find zero_shot_colab -type f | wc -l
```

**Olması gereken:** 50-150 dosya

**44000 ise:** Gereksiz klasörler var, temizleyin!

## 🎯 Sonuç:

**Sadece kaynak kodu yükleyin!**

- ✅ src/
- ✅ experiments/
- ✅ notebooks/
- ✅ main.py
- ✅ requirements.txt

**Geri kalanı gereksiz!**

Models, cache, git → Colab'da kurar/indirir.

**Beklenen boyut:** 1-5 MB, 50-100 dosya 🎯