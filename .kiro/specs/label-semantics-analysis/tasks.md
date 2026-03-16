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
  - [x] 5.2 Pilot sonuçlarını karşılaştıran küçük bir script yaz: `scripts/pilot_summary.py`
    - `results/` altındaki AG News × 3 model JSON'larını oku (L1, L2, L3)
    - `experiment_name`, `label_mode`, `macro_f1`, `accuracy` sütunlarıyla tablo yazdır
    - _Requirements: 2.3, 7.2_

- [x] 6. L3 labels — Kalan 8 dataset
  - `src/labels.py`'deki her dataset'e `multi_description` key'ini ekle
  - Dataset başına her label için 3 paraphrase description yaz
  - Sıra: `dbpedia_14`, `yahoo_answers_topics`, `banking77`, `zeroshot/twitter-financial-news-sentiment`, `SetFit/20_newsgroups`, `imdb`, `sst2`, `go_emotions`
  - _Requirements: 1.1, 1.7_

- [x] 7. Full-scale YAML config'leri — 9 dataset × 7 model × L3
  - [x] 7.1 `scripts/generate_configs.py` scripti yaz (veya mevcut varsa güncelle)
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

- [x] 9. Label Separability Analyzer — `scripts/analyze_separability.py`
  - [x] 9.1 `LabelSeparabilityAnalyzer` sınıfını yaz
    - `compute_similarity_matrix(label_dict, encoder)` → cosine similarity matrisi
    - `compute_separability_score(sim_matrix)` → off-diagonal ortalaması
    - L1, L2, L3 için ayrı ayrı hesapla; label sayısı ≤ 1 olan dataset'leri uyarıyla atla
    - _Requirements: 4.1, 4.2, 4.3, 4.6_
  - [x] 9.2 Separability × Macro-F1 Pearson korelasyonu hesapla
    - `results/` JSON'larından ortalama Macro-F1 oku
    - `scipy.stats.pearsonr` ile korelasyon hesapla
    - _Requirements: 4.7_
  - [x] 9.3 Sonuçları kaydet
    - `reports/separability/separability_scores.csv` — dataset, label_mode, separability_score sütunları
    - Dataset'leri separability_score'a göre sıralı yaz
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 4.4, 4.5, 7.2, 7.3, 7.6_
  - [ ]* 9.4 Separability hesaplama için property testi yaz
    - **Property 3: Identical labels → separability_score == 1.0** (tüm off-diagonal değerler 1)
    - **Property 4: Orthogonal labels → separability_score ≈ 0.0**
    - **Validates: Requirements 4.1, 4.2**

- [x] 10. Task × Label Interaction Analyzer — `scripts/analyze_task_interaction.py`
  - [x] 10.1 `TaskInteractionAnalyzer` sınıfını yaz
    - Dataset → task_type eşlemesini tanımla (requirements 5.5'teki mapping)
    - Her (task_type, label_mode) kombinasyonu için ortalama Macro-F1 hesapla
    - Eksik değerleri `NaN` olarak işaretle
    - _Requirements: 5.1, 5.5, 5.6_
  - [x] 10.2 ΔF1 hesapla ve görselleştir
    - Her dataset için `ΔF1(L2-L1)` ve `ΔF1(L3-L1)` hesapla
    - task_type × label_mode heatmap'i üret → `reports/task_interaction/heatmap.png` (≥300 DPI)
    - ΔF1 gain bar plot'u üret → `reports/task_interaction/delta_f1_gain.png` (≥300 DPI)
    - _Requirements: 5.2, 5.3, 5.4, 7.5_
  - [x] 10.3 Sonuçları CSV olarak kaydet
    - `reports/task_interaction/task_label_means.csv`
    - `reports/task_interaction/delta_f1.csv`
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 7.2, 7.3, 7.6_

- [x] 11. Model Sensitivity Analyzer — `scripts/analyze_model_sensitivity.py`
  - [x] 11.1 `ModelSensitivityAnalyzer` sınıfını yaz
    - Her model için tüm dataset'ler üzerinden `variance(F1_L1, F1_L2, F1_L3)` hesapla
    - Eksik label modu verisi olan modeller için uyarı logla, mevcut verilerle varyansı hesapla
    - _Requirements: 6.1, 6.3, 6.5_
  - [x] 11.2 Sensitivity tablosu ve görseli üret
    - Modelleri ortalama varyansa göre sıralayan tablo → `reports/model_sensitivity/sensitivity_scores.csv`
    - Bar chart → `reports/model_sensitivity/sensitivity_bar.png` (≥300 DPI)
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 6.2, 6.4, 7.2, 7.3, 7.5, 7.6_

- [ ] 12. Final checkpoint — Tüm analizler tamamlandı
  - Ensure all tests pass, ask the user if questions arise.

- [-] 13. LLM tabanlı label description üretim scripti — `scripts/generate_label_descriptions.py`
  - [-] 13.1 `DescriptionGenerator` sınıfını yaz
    - OpenAI GPT-4o ve Anthropic Claude 3.5 Sonnet API'lerini destekle
    - Sabit prompt şablonunu uygula: `"Define the following text classification label in 15-20 words, focusing only on its semantic core without using the label name itself. Dataset: [Dataset Name]. Label: [Label Name]."`
    - Tüm LLM çağrılarını `temperature=0` ile gerçekleştir
    - _Requirements: 8.1, 8.2_
  - [-] 13.2 Dataset başına otoriter kaynak eşlemesini uygula
    - `ag_news`, `yahoo_answers_topics`, `SetFit/20_newsgroups` → Wikipedia / Wikidata
    - `dbpedia_14` → DBpedia Ontology
    - `banking77` → dataset dokümantasyonu
    - `imdb`, `sst2`, `zeroshot/twitter-financial-news-sentiment` → psikoloji sözlüğü
    - `go_emotions` → Ekman / Plutchik teorisi
    - Kaynak erişilemezse `llm_fallback` kullan
    - _Requirements: 8.4, 8.7_
  - [ ]* 13.3 `DescriptionGenerator` için birim testi yaz
    - Mock API yanıtlarıyla prompt şablonunun doğru oluşturulduğunu doğrula
    - `temperature=0` parametresinin her çağrıda iletildiğini doğrula
    - **Validates: Requirements 8.1, 8.2**

- [ ] 14. Provenance ve generation metadata kaydı
  - [ ] 14.1 `src/label_descriptions/provenance.json` üretimini yaz
    - Her description için `dataset`, `label_id`, `label_mode`, `source_type`, `source_url_or_reference`, `generated_at` alanlarını kaydet
    - `source_type` değerleri: `llm_generated`, `wikipedia`, `wikidata`, `dbpedia_ontology`, `dataset_documentation`, `psychology_dictionary`, `ekman_theory`, `plutchik_theory`, `llm_fallback`
    - _Requirements: 8.5, 8.6_
  - [ ] 14.2 `src/label_descriptions/generation_metadata.json` üretimini yaz
    - Kullanılan model adı, prompt şablonu, temperature değeri ve üretim tarihini kaydet
    - _Requirements: 8.3_
  - [ ]* 14.3 Provenance kaydı için property testi yaz
    - **Property 5: Her provenance kaydı zorunlu 6 alanı içermeli** — `dataset`, `label_id`, `label_mode`, `source_type`, `source_url_or_reference`, `generated_at`
    - **Validates: Requirements 8.6**

- [ ] 15. `src/labels.py`'yi standart description'larla güncelle
  - [ ] 15.1 `LABEL_SETS`'e `description_set_a` (GPT-4o üretimi) key'ini ekle
    - 9 dataset için GPT-4o üretimi description'ları `description_set_a` altında tanımla
    - Mevcut `description` ve `multi_description` key'lerini değiştirme
    - _Requirements: 8.1, 9.1_
  - [ ] 15.2 `LABEL_SETS`'e `description_set_b` (Claude / sözlük kaynaklı) key'ini ekle
    - 9 dataset için Claude 3.5 Sonnet veya sözlük kaynaklı description'ları `description_set_b` altında tanımla
    - _Requirements: 9.1_
  - [ ]* 15.3 `LABEL_SETS` bütünlük testi yaz
    - Her dataset'in `description_set_a` ve `description_set_b` key'lerini içerdiğini doğrula
    - Her key altındaki label sayısının diğer modlarla tutarlı olduğunu doğrula
    - **Validates: Requirements 9.1**

- [ ] 16. Checkpoint — Description üretimi ve labels.py güncellemesi tamamlandı
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Robustness deneyi YAML config'leri — `experiments/robustness/`
  - [ ] 17.1 `scripts/generate_configs.py` scriptini `description_set_a` ve `description_set_b` modlarını kapsayacak şekilde güncelle
    - 9 dataset × 7 model × 2 set = 126 YAML dosyasını `experiments/robustness/` altına üret
    - Her YAML'da `label_mode: description_set_a` veya `label_mode: description_set_b` alanını ayarla
    - _Requirements: 9.2_
  - [ ]* 17.2 Config üretim testi yaz
    - Üretilen dosya sayısının 126 olduğunu doğrula
    - Her YAML'ın `label_mode` alanını içerdiğini doğrula
    - **Validates: Requirements 9.2**

- [ ] 18. Robustness Analyzer — `scripts/analyze_robustness.py`
  - [ ] 18.1 `RobustnessAnalyzer` sınıfını yaz
    - `results/` dizininden Set A ve Set B sonuçlarını oku
    - Her (dataset, model) çifti için `|ΔF1(A-B)|` hesapla
    - Tüm çiftler üzerinden ortalama `|ΔF1(A-B)|` hesapla; 0.02 eşiğiyle karşılaştır
    - Eksik Set B description'larını logla, mevcut verilerle analizi tamamla
    - _Requirements: 9.3, 9.4, 9.7_
  - [ ] 18.2 Robustness çıktılarını kaydet
    - `reports/robustness/robustness_scores.csv` — dataset, model, delta_f1_abs sütunları
    - `reports/robustness/robustness_heatmap.png` — dataset × model heatmap (≥300 DPI)
    - `reports/robustness/robustness_summary.md` — ortalama `|ΔF1(A-B)|`, en yüksek/düşük varyans gösteren çiftler
    - Hangi input/output dosyalarının kullanıldığını logla
    - _Requirements: 9.4, 9.5, 9.6, 7.3, 7.5, 7.6_
  - [ ]* 18.3 Robustness hesaplama için property testi yaz
    - **Property 6: Set A == Set B olduğunda |ΔF1(A-B)| == 0.0**
    - **Property 7: |ΔF1(A-B)| her zaman 0 ile 1 arasında olmalı**
    - **Validates: Requirements 9.3, 9.4**

- [ ] 19. Paper metodoloji dokümantasyonu — `reports/methodology/description_protocol.md`
  - Kullanılan LLM modelini, prompt şablonunu ve temperature değerini belgele
  - Her dataset için kaynak eşlemesini tablo formatında yaz
  - Paper'ın metodoloji bölümüne doğrudan dahil edilebilecek şekilde yaz
  - _Requirements: 8.8_

- [ ] 20. Final checkpoint — Robustness analizi tamamlandı
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- `*` ile işaretli sub-task'lar opsiyoneldir; MVP için atlanabilir
- Her task, traceability için requirements referansı içerir
- Tüm analiz scriptleri `results/` JSON'larından okur; experiment'ları yeniden çalıştırmaz (Req 7.2)
- `reports/` dizin yapısı: `separability/`, `task_interaction/`, `model_sensitivity/` (Req 7.3)
- Görseller ≥300 DPI PNG, tablolar CSV formatında kaydedilir (Req 7.5, 7.6)
- Mevcut `name_only` ve `description` modları hiçbir görevde değiştirilmez
