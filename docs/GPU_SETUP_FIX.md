# 🚀 GPU Setup - RTX 4060 Kullanımı

## ⚠️ ÖNEMLİ: Python Versiyon Sorunu

**Python 3.14.3 kullanıyorsunuz** - Bu çok yeni ve PyTorch henüz desteklemiyor!

PyTorch desteklenen Python versiyonları: **3.8, 3.9, 3.10, 3.11, 3.12**

## Çözüm Seçenekleri:

### Seçenek 1: Python 3.12 ile Yeni Sanal Ortam (ÖNERİLEN)

```bash
# Python 3.12 kur (python.org'dan)
# Sonra:
python3.12 -m venv .venv_gpu
.venv_gpu\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Seçenek 2: Conda Kullan (KOLAY)

```bash
# Conda kur (anaconda.com veya miniconda)
conda create -n zeroshot python=3.12
conda activate zeroshot
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Seçenek 3: PyTorch Nightly (RİSKLİ)

```bash
# Python 3.14 için PyTorch nightly build dene (garanti değil)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Durum

✅ **GPU:** NVIDIA GeForce RTX 4060 (8GB VRAM)  
✅ **Driver:** 581.83  
✅ **CUDA:** 13.0 destekli  
❌ **Python:** 3.14.3 (PyTorch desteklemiyor!)  
❌ **PyTorch:** 2.10.0+cpu (CPU-only)

## Önerilen Çözüm: Conda ile Kurulum

En kolay ve güvenilir yöntem **Conda**:

### 1️⃣ Miniconda İndir ve Kur
https://docs.conda.io/en/latest/miniconda.html

### 2️⃣ Yeni Ortam Oluştur
```bash
conda create -n zeroshot python=3.12 -y
conda activate zeroshot
```

### 3️⃣ PyTorch + CUDA Kur
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4️⃣ Diğer Paketleri Kur
```bash
pip install -r requirements.txt
```

### 5️⃣ Doğrula
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Beklenen:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 🎉 Sonuç

✅ Artık modeller GPU'da çalışacak  
✅ **10-50x daha hızlı** olacak  
✅ Python 3.12 stabil ve PyTorch tam destekliyor  

## 📊 Performans Karşılaştırması

| İşlem | CPU (Python 3.14) | RTX 4060 (Python 3.12) | Hızlanma |
|-------|-------------------|------------------------|----------|
| 1000 text encode (MPNet) | ~30s | ~2s | **15x** |
| 1000 text encode (Qwen3-8B) | ~5 min | ~20s | **15x** |
| 20 Newsgroups (2000 samples) | ~10 min | ~1 min | **10x** |

## ⚠️ Notlar

- Python 3.14 çok yeni, birçok paket desteklemiyor
- Python 3.12 ideal (stabil + modern)
- Conda PyTorch kurulumu için en kolay yol
- Mevcut `.venv` silinmeyecek, yeni ortam oluşturacaksınız

## 🔍 Sorun Giderme

### Hala CPU kullanıyorsa:
```bash
python scripts/check_gpu_and_fix.py
```

### Conda ortamı aktif mi kontrol:
```bash
conda env list
# * işareti aktif ortamı gösterir
```

### requirements.txt'teki torch conflict:
```bash
# requirements.txt'te torch satırını kaldırın veya yorum yapın
# Conda ile kurulumu sonra requirements.txt'i kurun