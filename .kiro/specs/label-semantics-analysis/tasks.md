# Implementation Plan: Label Semantics Analysis

## Overview

L3 (`multi_description`) label modunu mevcut pipeline'a entegre eder, pilot ve full-scale deneyleri çalıştırır, ardından separability, task interaction ve model sensitivity analizlerini üretir. Tüm değişiklikler additive'dir; mevcut `name_only` ve `description` modları korunur.

## Tasks

- [x] 1. L3 multi_description labels — AG News (Pilot)
  - `src/labels.py` içindeki `ag_news` entry'sine `multi_description` key'ini ekle
  - Her label için tam olarak 3 paraphrase description yaz (tasarım dokümanındaki AG News örneklerini kullan)
  - _Requirements: 1.1, 1.7_

- [x] 2. build_multi_description_embeddings fonksiyonu
  - [x] 2.1 `src/labels.py`'ye `build_multi_description_embeddings` fonksiyonunu ekle
    - `label_dict: Dict[int, List[str]]`, `encoder: BiEncoder`, `normalize: bool`, `batch_size: int` parametreleri
    - Her label için 3 description'ı encode et, mean pooling uygula
    - `(label_embeddings: np.ndarray, label_ids: List[int])` döndür — shape: `(num_classes, dim)`
    - _Requirements: 1.3_
  - [ ]* 2.2 `build_multi_description_embeddings` için property testi yaz
    - **Property 1: Output shape correctness** — `label_embeddings.shape == (num_classes, dim)`
    - **Property 2: Mean pooling idempotency** — 3 özdeş description verildiğinde sonuç tek description embedding'ine eşit olmalı
    - **Validates: Requirements 1.3**

- [x] 3. Runner'a multi_description dalı ekle
  - [x] 3.1 `src/runner.py`'deki biencoder pipeline bloğunu güncelle
    - `label_mode == "multi_description"` kontrolü ekle
    - `build_multi_description_embeddings` çağır, dönen `label_emb` ile direkt similarity hesapla (`predict_biencoder` yerine)
    - `else` dalında mevcut `flatten_label_texts → predict_biencoder` akışı değişmeden kalmalı
    - _Requirements: 1.2, 1.3, 1.5_
  - [x] 3.2 Hata yönetimi ekle
    - `multi_description` key'i `LABEL_SETS`'te yoksa açıklayıcı `ValueError` fırlat
    - _Requirements: 1.6_
  - [ ]* 3.3 Runner entegrasyon testi yaz
    - Mock encoder ile `label_mode: multi_description` çalıştır, metadata'da `label_mode` alanını doğrula
    - `name_only` ve `description` modlarının hâlâ çalıştığını doğrula (regression)
    - **Validates: Requirements 1.2, 1.5**

- [ ] 4. Checkpoint — Pilot öncesi doğrulama
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Pilot YAML config'leri — AG News × 3 model × L3
  - [x] 5.1 `experiments/label_formulation/` altında AG News × `multi_description` için 3 YAML dosyası oluştur
    - Modeller: `INSTRUCTOR-large`, `bge-m3`, `all-mpnet-base-v2`
    - Mevcut `name_only` / `description` config'lerini şablon olarak kullan; yalnızca `label_mode` ve `experiment_name` değiştir
    - _Requirements: 2.1_
  - [ ] 5.2 Pilot sonuçlarını karşılaştıran küçük bir script yaz: `scripts/pilot_summary.py`
    - `results/` altındaki AG News × 3 model JSON'larını oku (L1, L2, L3)
    - `experiment_name`, `label_mode`, `macro_f1`, `accuracy` sütunlarıyla tablo yazdır
    - _Requirements: 2.3, 7.2_

- [ ] 6. L3 labels — Kalan 8 dataset
  - `src/labels.py`'deki her dataset'e `multi_description` key'ini ekle
  - Dataset başına her label için 3 paraphrase description yaz
  - Sıra: `dbpedia_14`, `yahoo_answers_topics`, `banking77`, `zeroshot/twitter-financial-news-sentiment`, `SetFit/20_newsgroups`, `imdb`, `sst2`, `go_emotions`
  - _Requirements: 1.1, 1.7_

- [ ] 7. Full-scale YAML config'leri — 9 dataset × 7 model × L3
  - [ ] 7.1 `scripts/generate_configs.py` scripti yaz (veya mevcut varsa güncelle)
    - 9 dataset × 7 model kombinasyonu için `multi_description` YAML dosyaları üret
    - 7 model: `INSTRUCTOR-large`, `bge-m3`, `all-mpnet-base-v2`, `nomic-embed-text`, `e5-large-v2`, `jina-embeddings-v3`, `qwen3-embedding`
    - Toplam 63 dosya `experiments/label_formulation/` altına yazılmalı
    - _Requirements: 3.1, 3.2_
  - [ ]* 7.2 Config üretim testi yaz
    - Üretilen dosya sayısının 63 olduğunu doğrula
    - Her YAML'ın gerekli alanları (`experiment_name`, `dataset.name`, `task.label_mode`, `models.biencoder.name`) içerdiğini doğrula
    - **Validates: Requirements 3.1, 3.2**

- [ ] 8. Checkpoint — Full-scale öncesi doğrulama
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Label Separability Analyzer — `scripts/analyze_separability.py`
  - [ ] 9.1 `LabelSeparabilityAnalyzer` sınıfını yaz
    - `compute_similarity_matrix(label_dict, encoder)` → cosine similarity matrisi
    - `compute_separability_score(sim_matrix)` → off-diagonal ortalaması
    - L1, L2, L3 için ayrı ayrı hesapla; label sayısı ≤ 1 olan dataset'leri uyarıyla atla
    - _Requirements: 4.1, 4.2, 4.3, 4.6_
  - [ ] 9.2 Separability × Macro-F1 Pearson korelasyonu hesapla
    - `results/` JSON'larından ortalama Macro-F1 oku
    - `scipy.stats.pearsonr` ile korelasyon hesapla
    - _Requirements: 4.7_
  - [ ] 9.3 Sonuçları kaydet
    - `reports/separability/separability_scores.csv` — dataset, label_mode, separability_score sütunları
    - Dataset'leri separability_score'a göre sıralı yaz
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 4.4, 4.5, 7.2, 7.3, 7.6_
  - [ ]* 9.4 Separability hesaplama için property testi yaz
    - **Property 3: Identical labels → separability_score == 1.0** (tüm off-diagonal değerler 1)
    - **Property 4: Orthogonal labels → separability_score ≈ 0.0**
    - **Validates: Requirements 4.1, 4.2**

- [ ] 10. Task × Label Interaction Analyzer — `scripts/analyze_task_interaction.py`
  - [ ] 10.1 `TaskInteractionAnalyzer` sınıfını yaz
    - Dataset → task_type eşlemesini tanımla (requirements 5.5'teki mapping)
    - Her (task_type, label_mode) kombinasyonu için ortalama Macro-F1 hesapla
    - Eksik değerleri `NaN` olarak işaretle
    - _Requirements: 5.1, 5.5, 5.6_
  - [ ] 10.2 ΔF1 hesapla ve görselleştir
    - Her dataset için `ΔF1(L2-L1)` ve `ΔF1(L3-L1)` hesapla
    - task_type × label_mode heatmap'i üret → `reports/task_interaction/heatmap.png` (≥300 DPI)
    - ΔF1 gain bar plot'u üret → `reports/task_interaction/delta_f1_gain.png` (≥300 DPI)
    - _Requirements: 5.2, 5.3, 5.4, 7.5_
  - [ ] 10.3 Sonuçları CSV olarak kaydet
    - `reports/task_interaction/task_label_means.csv`
    - `reports/task_interaction/delta_f1.csv`
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 7.2, 7.3, 7.6_

- [ ] 11. Model Sensitivity Analyzer — `scripts/analyze_model_sensitivity.py`
  - [ ] 11.1 `ModelSensitivityAnalyzer` sınıfını yaz
    - Her model için tüm dataset'ler üzerinden `variance(F1_L1, F1_L2, F1_L3)` hesapla
    - Eksik label modu verisi olan modeller için uyarı logla, mevcut verilerle varyansı hesapla
    - _Requirements: 6.1, 6.3, 6.5_
  - [ ] 11.2 Sensitivity tablosu ve görseli üret
    - Modelleri ortalama varyansa göre sıralayan tablo → `reports/model_sensitivity/sensitivity_scores.csv`
    - Bar chart → `reports/model_sensitivity/sensitivity_bar.png` (≥300 DPI)
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 6.2, 6.4, 7.2, 7.3, 7.5, 7.6_

- [ ] 12. Final checkpoint — Tüm analizler tamamlandı
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- `*` ile işaretli sub-task'lar opsiyoneldir; MVP için atlanabilir
- Her task, traceability için requirements referansı içerir
- Tüm analiz scriptleri `results/` JSON'larından okur; experiment'ları yeniden çalıştırmaz (Req 7.2)
- `reports/` dizin yapısı: `separability/`, `task_interaction/`, `model_sensitivity/` (Req 7.3)
- Görseller ≥300 DPI PNG, tablolar CSV formatında kaydedilir (Req 7.5, 7.6)
- Mevcut `name_only` ve `description` modları hiçbir görevde değiştirilmez
