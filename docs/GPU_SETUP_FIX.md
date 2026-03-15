# 🚀 GPU Setup - RTX 4060 Kullanımı

## Sorun

PyTorch **CPU-only** versiyonu kurulu (`2.10.0+cpu`), bu yüzden RTX 4060'ınız kullanılmıyor.

## Durum

✅ **GPU:** NVIDIA GeForce RTX 4060 (8GB VRAM)  
✅ **Driver:** 581.83  
✅ **CUDA:** 13.0 destekli  
❌ **PyTorch:** CPU-only (2.10.0+cpu)

## Çözüm - 3 Adım

### 1️⃣ Mevcut PyTorch'u Kaldır

```bash
pip uninstall torch torchvision torchaudio
```

Tüm uyarıları onaylayın (y)

### 2️⃣ CUDA-Enabled PyTorch Kur

**CUDA 12.1 için (önerilen):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**VEYA CUDA 11.8 için:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3️⃣ Doğrula

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**Beklenen çıktı:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 🎉 Sonuç

✅ Artık modeller GPU'da çalışacak  
✅ **10-50x daha hızlı** olacak  
✅ Kod değişikliği **gerekmez** (otomatik GPU kullanılır)  
✅ Batch size daha büyük kullanılabilir

## 📊 Performans Karşılaştırması

| İşlem | CPU | RTX 4060 | Hızlanma |
|-------|-----|----------|----------|
| 1000 text encode (MPNet) | ~30s | ~2s | **15x** |
| 1000 text encode (Qwen3-8B) | ~5 min | ~20s | **15x** |
| 20 Newsgroups (2000 samples) | ~10 min | ~1 min | **10x** |

## ⚠️ Notlar

- CUDA 12.1 PyTorch yeni sürümleri destekler
- CUDA 11.8 eski GPU'larla daha uyumlu
- RTX 4060 her ikisiyle de çalışır
- requirements.txt'i değiştirmeye gerek yok (sadece PyTorch)

## 🔍 Sorun Giderme

### "CUDA out of memory" hatası alıyorsanız:
- `batch_size=8` zaten ayarlanmış (label formulation için)
- Daha da düşürmeniz gerekirse: `batch_size=4`
- Model boyutlarını kontrol edin (Qwen3-8B çok büyük)

### GPU kullanılmıyor gibi görünüyorsa:
```bash
python scripts/check_gpu_and_fix.py
```

Bu script GPU durumunu kontrol eder ve sorunları gösterir.