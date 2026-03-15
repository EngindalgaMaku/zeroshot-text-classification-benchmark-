# 🚀 GPU Kurulum - Adım Adım Rehber

## Durum

Şu an `.venv` sanal ortamında Python 3.14.3 kullanıyorsunuz.  
PyTorch Python 3.14'ü desteklemiyor, bu yüzden GPU kullanamıyorsunuz.

## En Kolay Çözüm: Conda (Önerilen)

Conda kullanarak **yeni bir ortam** oluşturacağız. `.venv` silinmeyecek, olduğu gibi kalacak.

---

## Adım 1: Miniconda İndir ve Kur

### 1.1 İndir:
https://docs.conda.io/en/latest/miniconda.html

**Windows için:** `Miniconda3 Windows 64-bit` 

### 1.2 Kur:
- İndirdiğiniz `.exe` dosyasını çalıştırın
- "Next" → "I Agree" → **"Just Me"** seçin
- Varsayılan klasörde kurulabilir
- ✅ **"Add Miniconda3 to PATH"** kutucuğunu işaretleyin (önemli!)
- Install → Finish

### 1.3 Test et:
**YENİ** PowerShell/Terminal açın (eski kapatıp yeni açın!)

```bash
conda --version
```

Versiyon görüyorsanız ✅ kurulum başarılı!

---

## Adım 2: Conda Ortamı Oluştur

**YENİ terminal'de:**

```bash
# Proje klasörüne git
cd "C:\Users\Engin Dalga\Documents\GitHub\zeroshot_nlp__new"

# Conda ortamı oluştur (Python 3.12 ile)
conda create -n zeroshot python=3.12 -y
```

Bu işlem 1-2 dakika sürebilir. Paketleri indirecek.

---

## Adım 3: Conda Ortamını Aktive Et

```bash
conda activate zeroshot
```

Şimdi terminal'de başta `(zeroshot)` yazısı görünmeli:
```
(zeroshot) PS C:\Users\Engin Dalga\Documents\GitHub\zeroshot_nlp__new>
```

---

## Adım 4: PyTorch + CUDA Kur

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Bu işlem 5-10 dakika sürebilir. **Büyük indirme** (~2GB).

---

## Adım 5: Diğer Paketleri Kur

```bash
pip install -r requirements.txt
```

2-3 dakika sürer.

---

## Adım 6: GPU'yu Test Et

```bash
python scripts/check_gpu_and_fix.py
```

**Görmek istediğiniz:**
```
✅ CUDA available: True
✅ GPU device: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Adım 7: Artık Her Zaman Conda Kullanın

### VS Code'da:
1. Ctrl+Shift+P → "Python: Select Interpreter"
2. "zeroshot (conda)" seçin

### Terminal'de:
```bash
conda activate zeroshot
```

Her seferinde conda ortamını aktive etmeyi unutmayın!

---

## ❓ Sık Sorulan Sorular

### Q: `.venv` ne olacak?
**A:** Silinmeyecek, olduğu gibi kalacak. Artık conda ortamını kullanacaksınız.

### Q: Conda her seferinde aktive etmem mi lazım?
**A:** Evet, yeni terminal açtığınızda `conda activate zeroshot` yapın.

### Q: VS Code otomatik aktive etsin mi?
**A:** Evet! "Python: Select Interpreter" ile conda ortamını seçin, otomatik aktive olur.

### Q: Eski .venv'e geri dönebilir miyim?
**A:** Evet:
```bash
conda deactivate
.venv\Scripts\activate
```

### Q: Conda çok yer kaplıyor mu?
**A:** ~3-4 GB (PyTorch + CUDA büyük). Ama GPU kullanmak için gerekli.

---

## 🎉 Artık GPU Hazır!

Deneyleri çalıştırdığınızda:
- ✅ GPU otomatik kullanılacak
- ✅ 10-50x daha hızlı olacak
- ✅ Kod değişikliği gerektirmez

```bash
# Test için basit bir deney çalıştırın
conda activate zeroshot
python main.py --config experiments/exp_agnews_mpnet.yaml
```

İlk satırlarda "Model loaded on device: cuda:0" gibi bir mesaj göreceksiniz!

---

## 🔧 Sorun mu Yaşıyorsunuz?

### "conda: command not found"
- Miniconda'yı yeniden kurun
- "Add to PATH" seçeneğini işaretleyin
- Terminal'i kapatıp yeniden açın

### "CUDA: False" gösteriyorsa
```bash
python scripts/check_gpu_and_fix.py
```
Bu script sorunu gösterecek.

### VS Code conda görmüyorsa
- Ctrl+Shift+P → "Python: Select Interpreter"
- "Refresh" butonuna basın
- "zeroshot (conda)" görünmeli