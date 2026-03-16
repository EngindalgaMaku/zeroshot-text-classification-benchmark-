# 📱 Google Colab Kullanım Rehberi

## 🎯 Temel Mantık

**Önemli:** Notebook'lar (.ipynb) **sadece interface**'dir. Asıl kod **hala .py dosyalarında**!

### Nasıl Çalışıyor?

```
Google Drive:
  └── zero_shot_reliable_cls/     ← TÜM proje buraya
      ├── src/                    ← Python kodları
      │   ├── data.py
      │   ├── labels.py
      │   └── ...
      ├── main.py                 ← Ana script
      ├── experiments/            ← Config dosyaları
      └── notebooks/              ← Interface
          └── 01_run_experiments.ipynb  ← Bu sadece komut çalıştırıyor!
```

**Notebook ne yapar?**
- Drive'ı mount eder
- Proje klasörüne gider
- `!python main.py ...` komutunu çalıştırır
- **Arka planda .py dosyaları çalışır!**

## 📋 Adım Adım Colab Kurulumu

### 1. Proje Klasörünü Drive'a Yükle

**İki yöntem:**

#### Yöntem A: Sürükle-Bırak (Kolay)
1. Google Drive'ı aç: https://drive.google.com
2. "MyDrive" klasörüne gir
3. **TÜM proje klasörünü** (zero_shot_reliable_cls) sürükle-bırak
4. Yükleme tamamlanmasını bekle (~5-10 dakika)

#### Yöntem B: Sync ile (Otomatik)
1. Google Drive Desktop uygulamasını kur
2. Proje klasörünü Drive klasörüne kopyala
3. Otomatik senkronize olur

**ÖNEMLİ:** TÜM proje klasörü yüklenecek:
```
✅ src/ klasörü
✅ main.py
✅ experiments/ klasörü
✅ notebooks/ klasörü
✅ requirements.txt
✅ HER ŞEY!
```

### 2. Colab'da Notebook'u Aç

1. **Drive'da notebook'a sağ tıkla:**
   ```
   MyDrive/zero_shot_reliable_cls/notebooks/01_run_experiments.ipynb
   ```

2. **"Open with" → "Google Colaboratory"**

3. **İlk kez kullanıyorsanız:** "Google Colaboratory" uygulamasını ekleyin

### 3. GPU'yu Aktif Et

Colab'da:
1. Üstte **"Runtime"** menüsüne tıkla
2. **"Change runtime type"** seç
3. **"Hardware accelerator"** → **"GPU"** seç
4. **"Save"**

### 4. Hücreleri Çalıştır

Notebook'ta sırayla çalıştır:

#### Hücre 1: Drive'ı Mount Et
```python
from google.colab import drive
drive.mount('/content/drive')
```
→ İzin ver

#### Hücre 2: Proje Klasörüne Git
```python
PROJECT_PATH = '/content/drive/MyDrive/zero_shot_reliable_cls'
%cd {PROJECT_PATH}
!pwd
!ls -la
```
→ Dosyalar görünmeli

#### Hücre 3: Requirements Kur
```python
!pip install -q -r requirements.txt
```
→ 2-3 dakika sürer

#### Hücre 4: GPU Kontrolü
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
→ "True" ve GPU ismi görünmeli

#### Hücre 5: Deney Çalıştır
```python
!python main.py --config experiments/exp_agnews_hybrid.yaml
```
→ **ÖNEMLİ:** Bu komut main.py'yi çalıştırır, o da src/ klasöründeki kodları kullanır!

## 🔍 Ne Oluyor Arka Planda?

### Notebook'tan Komut Çalışınca:

```
Notebook: !python main.py --config ...
  ↓
main.py çalışır
  ↓
src/runner.py import edilir
  ↓
src/encoders.py, src/pipeline.py vb. kullanılır
  ↓
Sonuçlar results/raw/ klasörüne kaydedilir
```

**Yani:**
- Notebook sadece **komutları çalıştırıyor**
- Asıl kod **hala .py dosyalarında**
- Hiçbir fark yok, sadece **GPU var** artık!

## 💡 Avantajlar

### Neden Colab?

1. **Ücretsiz GPU** 🎁
   - CPU: 10 dakika
   - GPU: 2 dakika
   - 5x hızlanma!

2. **Aynı Kod** ✅
   - Hiçbir değişiklik yok
   - Sadece daha hızlı çalışıyor

3. **Her Yerden Erişim** 🌐
   - Evden, okuldan, her yerden
   - Tarayıcı yeterli

## 🚀 Pratik Kullanım

### Tipik Bir Colab Workflow:

```python
# 1. Drive mount
from google.colab import drive
drive.mount('/content/drive')

# 2. Klasöre git
%cd /content/drive/MyDrive/zero_shot_reliable_cls

# 3. Deney çalıştır
!python main.py --config experiments/exp_agnews_hybrid.yaml

# 4. Sonuçları oku
import json
with open("results/raw/agnews_bge_bge_reranker_metrics.json") as f:
    metrics = json.load(f)
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

## ❓ SSS

### "Notebook her şeyi içeriyor mu?"

HAYIR! Notebook sadece komutları çalıştırıyor:
- ✅ `!python main.py ...` → main.py'yi çalıştır
- ✅ `!pip install ...` → paketleri kur
- ✅ Sonuçları göster

Asıl kod hala `.py` dosyalarında!

### "Her seferinde requirements kurmam gerekir mi?"

EVET, çünkü Colab her oturumda sıfırdan başlar. Ama:
- Cache'lenir, 2. sefer daha hızlı
- GPU workspace kalıcı olur

### "Kodları değiştirirsem ne olur?"

1. Lokal editörde (VS Code) değiştir
2. Drive'a yükle (sync olursa otomatik)
3. Colab'da tekrar çalıştır
4. Yeni kod kullanılır!

### "Drive'daki değişiklikler otomatik görünür mü?"

EVET! Drive mount edilince canlı bağlantı var:
- Lokal'de değiştir → Drive'da güncellenir → Colab'da görünür

### "Kaç saat kullanabilirim?"

**Ücretsiz tier:**
- ~12 saat oturum
- Sonra disconnect
- Yeniden başlat, devam et

**Colab Pro ($10/ay):**
- 24 saat
- Daha güçlü GPU
- Daha az kesinti

## 🎯 Sonuç

**Colab kullanımı çok basit:**

1. ✅ Proje klasörünü Drive'a yükle
2. ✅ Notebook'u Colab'da aç
3. ✅ GPU seç
4. ✅ Hücreleri çalıştır
5. ✅ **Kod aynı, sadece GPU'da çalışıyor!**

**Hiçbir kod değişikliği gerekmez!** Sadece daha hızlı çalışır. 🚀

## 📝 İlk Kez Deneyenler İçin

```python
# Tek Tek Çalıştır:

# 1. Mount
from google.colab import drive
drive.mount('/content/drive')

# 2. Git
%cd /content/drive/MyDrive/zero_shot_reliable_cls

# 3. List
!ls -la

# 4. Check GPU
import torch
print(torch.cuda.is_available())

# 5. Install
!pip install -q -r requirements.txt

# 6. Run!
!python main.py --config experiments/exp_agnews_baseline.yaml
```

**Bu kadar!** 5 dakikada kurulu, çalışıyor. ✨