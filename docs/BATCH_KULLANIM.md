# 🚀 Otomatik Klasör Hazırlama

## Windows Kullanıcıları için

### Tek Tıkla Hazırla! ✨

1. **prepare_for_colab.bat** dosyasına çift tıklayın
2. Bekleyin (5-10 saniye)
3. **zero_shot_colab** klasörü oluşur
4. Bu klasörü Drive'a yükleyin!

### Ne Yapar?

✅ Gerekli dosyaları kopyalar:
- src/ klasörü
- experiments/ klasörü
- notebooks/ klasörü
- main.py
- requirements.txt
- README.md

✅ Gereksizleri temizler:
- __pycache__
- *.pyc
- .DS_Store

✅ Sonuç klasörleri oluşturur:
- results/raw/
- results/tables/
- results/plots/
- data_cache/

✅ Dosya sayısını kontrol eder:
- 50-500: ✅ Normal
- 500+: ⚠️ Uyarı verir

## Mac/Linux Kullanıcıları için

### Terminal'de:

```bash
# İzin ver
chmod +x prepare_for_colab.sh

# Çalıştır
./prepare_for_colab.sh
```

## 📊 Beklenen Sonuç

```
zero_shot_colab/               ← ~50-100 dosya
├── src/                       ← Python kodları
├── experiments/               ← Config dosyaları
├── notebooks/                 ← Colab notebooks
├── main.py
├── requirements.txt
├── README.md
└── results/                   ← Boş klasörler
    ├── raw/
    ├── tables/
    └── plots/
```

**Boyut:** 1-5 MB  
**Dosya sayısı:** 50-150  
**Yükleme süresi:** 1-2 dakika

## 🔄 Güncellemeler İçin

**Kodları değiştirdiniz mi?**

1. **prepare_for_colab.bat** dosyasına tekrar çift tıklayın
2. Yeni **zero_shot_colab** klasörü oluşur (eski silinir)
3. Drive'a yeniden yükleyin

**Veya Drive Desktop kullanıyorsanız:**
- Drive klasöründeki **zero_shot_colab** içine değişiklikleri kopyalayın
- Otomatik senkronize olur

## ⚡ Hızlı Workflow

### İlk Kez:
1. `prepare_for_colab.bat` çalıştır
2. `zero_shot_colab` → Drive'a yükle
3. Colab'da aç, GPU seç, çalıştır

### Sonraki Güncellemeler:
1. Lokal'de kod değiştir
2. `prepare_for_colab.bat` çalıştır
3. Drive'a yeniden yükle (üzerine yaz)

## 🎯 Avantajlar

**Otomatik:**
- ✅ Tek tıkla hazır
- ✅ Gereksizler temizlenir
- ✅ Dosya sayısı kontrol edilir

**Güvenli:**
- ✅ Orijinal proje dokunulmaz
- ✅ Yeni klasör oluşturur
- ✅ Eski klasör silinir (çakışma olmaz)

**Hızlı:**
- ✅ 5-10 saniye
- ✅ Manuel kopyalamadan daha hızlı
- ✅ Hata riski yok

## ❓ Sorun Giderme

### "Dosya sayısı hala çok fazla"

Kontrol edin:
```
zero_shot_colab/
  ├── .git/        ← OLMAMALI
  ├── venv/        ← OLMAMALI
  ├── __pycache__/ ← OLMAMALI
```

Çözüm: Batch dosyasını kontrol edin veya manuel silin.

### "Bat dosyası çalışmıyor"

1. Proje klasöründe olduğunuzdan emin olun
2. Windows Explorer'dan çift tıklayın
3. Veya cmd'de: `prepare_for_colab.bat`

### Mac/Linux'ta izin hatası

```bash
chmod +x prepare_for_colab.sh
./prepare_for_colab.sh
```

## 💡 İpuçları

**Drive Desktop kullanıyorsanız:**
```
Google Drive/zero_shot_colab/
```
Bu klasörü Drive Desktop'ta tutun, otomatik senkronize olur!

**Git kullanıyorsanız:**
Batch dosyası zaten .git/ klasörünü kopyalamaz, sorun yok!

**Colab Pro kullanıyorsanız:**
Aynı klasörü kullanın, sadece daha hızlı çalışır!

## ✅ Sonuç

**Artık her güncelleme için:**

1. Kodu değiştir
2. `prepare_for_colab.bat` çalıştır
3. Drive'a yükle
4. Colab'da çalıştır

**5 dakikada hazır!** 🚀