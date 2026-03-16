# Requirements Document

## Introduction

Bu feature, mevcut zero-shot text classification benchmark projesine kapsamlı bir "Label Semantics Analysis" katmanı ekler. Araştırma sorusu şudur: "How do label semantics and task structure influence the effectiveness of sentence embeddings in zero-shot text classification?"

Mevcut altyapı L1 (`name_only`) ve L2 (`description`) label modlarını desteklemektedir. Bu feature; L3 (`multi_description`) label modunu, label separability analizini, task × label interaction analizini ve model sensitivity analizini sisteme entegre eder. Tüm analizler mevcut `LABEL_SETS` yapısı, YAML config formatı ve BiEncoder pipeline'ı korunarak gerçekleştirilir.

---

## Glossary

- **L1**: `name_only` label modu — her sınıf için yalnızca sınıf adı kullanılır.
- **L2**: `description` label modu — her sınıf için tek cümlelik açıklama kullanılır.
- **L3**: `multi_description` label modu — her sınıf için 3 farklı paraphrase/description kullanılır; embedding'leri mean pooling ile birleştirilir.
- **Label_Embedding**: Bir label metninin veya birden fazla label metninin mean pooling sonucu elde edilen vektör temsili.
- **Label_Separability**: Bir dataset'teki label embedding'leri arasındaki ortalama off-diagonal cosine similarity değeri; düşük değer daha iyi ayrışabilirliği gösterir.
- **Mean_Pooling**: Birden fazla embedding vektörünün eleman bazında ortalaması alınarak tek bir vektör elde edilmesi işlemi.
- **LABEL_SETS**: `src/labels.py` içindeki, dataset adından label moduna ve label ID'den metin listesine eşleyen sözlük yapısı.
- **Pipeline**: `src/pipeline.py` içindeki BiEncoder tabanlı zero-shot classification akışı.
- **Runner**: `src/runner.py` içindeki, YAML config'den experiment çalıştıran modül.
- **Macro_F1**: Sınıf başına F1 skorlarının ağırlıksız ortalaması; sınıf dengesizliğine karşı robust bir metrik.
- **ΔF1**: İki label modu arasındaki Macro-F1 farkı (örn. ΔF1(L2-L1) = F1_L2 − F1_L1).
- **Task_Type**: Dataset'in sınıflandırma görevi türü: `topic`, `sentiment`, `intent`, `emotion`.
- **Model_Sensitivity**: Bir modelin L1/L2/L3 label modları arasındaki Macro-F1 varyansı.
- **Separability_Score**: Bir dataset için hesaplanan Label_Separability değeri (0–1 arası; düşük = daha ayrışık).
- **Pilot_Dataset**: AG News — 4 sınıf, topic görevi, L3 pipeline doğrulaması için kullanılan ilk dataset.
- **BiEncoder**: `src/encoders.py` içindeki, metin ve label'ları ayrı ayrı encode eden model.

---

## Requirements

### Requirement 1: L3 Label Modu (Multi-Description)

**User Story:** Araştırmacı olarak, her sınıf için birden fazla paraphrase description kullanmak istiyorum; böylece tek bir description'ın yetersiz kaldığı durumlarda label representation kalitesini artırabileyim.

#### Acceptance Criteria

1. THE `LABEL_SETS` dict'i, mevcut `name_only` ve `description` key'lerine ek olarak her dataset için `multi_description` key'ini içermelidir; bu key altında her label ID, tam olarak 3 string içeren bir liste ile eşlenmelidir.
2. WHEN `label_mode: multi_description` olarak ayarlandığında, THE `Runner` SHALL `get_label_texts` fonksiyonunu çağırarak 3 description'ı döndürmelidir.
3. WHEN `label_mode: multi_description` olarak ayarlandığında, THE `Pipeline` SHALL her label için 3 description'ın embedding'lerini ayrı ayrı hesaplamalı ve mean pooling uygulayarak tek bir `Label_Embedding` üretmelidir.
4. THE `flatten_label_texts` fonksiyonu, `multi_description` modunda mean pooling yapacak şekilde güncellenmelidir; bu güncelleme L1 ve L2 modlarının mevcut davranışını değiştirmemelidir.
5. WHEN `label_mode: multi_description` ile bir experiment çalıştırıldığında, THE `Runner` SHALL sonuç metadata'sına `label_mode: multi_description` değerini kaydetmelidir.
6. IF bir dataset için `multi_description` key'i `LABEL_SETS`'te tanımlı değilse, THEN THE `Runner` SHALL açıklayıcı bir hata mesajı fırlatmalıdır.
7. THE `multi_description` label'ları, tüm 9 dataset için yazılmalıdır: `ag_news`, `dbpedia_14`, `yahoo_answers_topics`, `banking77`, `zeroshot/twitter-financial-news-sentiment`, `SetFit/20_newsgroups`, `imdb`, `sst2`, `go_emotions`.

---

### Requirement 2: Pilot Experiment (AG News × 3 Model)

**User Story:** Araştırmacı olarak, L3 pipeline'ını tam ölçeğe geçmeden önce küçük bir pilot ile doğrulamak istiyorum; böylece hataları erken tespit edebilir ve zaman kaybını önleyebilirim.

#### Acceptance Criteria

1. THE `experiments/label_formulation/` dizini, AG News × {`INSTRUCTOR-large`, `bge-m3`, `all-mpnet-base-v2`} × {`name_only`, `description`, `multi_description`} kombinasyonları için toplam 9 YAML config dosyası içermelidir.
2. WHEN pilot experiment'lar çalıştırıldığında, THE `Runner` SHALL her experiment için `results/` altına metrics ve predictions kaydetmelidir.
3. WHEN pilot sonuçları elde edildiğinde, THE `Runner` SHALL L1, L2 ve L3 için Macro-F1 değerlerini karşılaştırılabilir formatta raporlamalıdır.
4. IF pilot experiment'lardan herhangi biri başarısız olursa, THEN THE `Runner` SHALL hata mesajını ve başarısız olan experiment adını loglamalıdır; diğer experiment'ların çalışmasını durdurmamalıdır.

---

### Requirement 3: Full-Scale Experiment (9 Dataset × 7 Model)

**User Story:** Araştırmacı olarak, pilot doğrulamasının ardından tüm dataset ve model kombinasyonlarında L3 deneylerini çalıştırmak istiyorum; böylece kapsamlı karşılaştırmalı analiz yapabileyim.

#### Acceptance Criteria

1. THE `experiments/label_formulation/` dizini, 9 dataset × 7 model × `multi_description` label modu için toplam 63 YAML config dosyası içermelidir.
2. THE 7 model şunlardır: `INSTRUCTOR-large`, `bge-m3`, `all-mpnet-base-v2`, `nomic-embed-text`, `e5-large-v2`, `jina-embeddings-v3`, `qwen3-embedding`.
3. WHEN full-scale experiment'lar çalıştırıldığında, THE `Runner` SHALL `skip_existing: true` parametresini desteklemeli ve tamamlanmış experiment'ları yeniden çalıştırmamalıdır.
4. THE her experiment sonucu, `experiment_name`, `dataset`, `label_mode`, `biencoder`, `macro_f1`, `accuracy` alanlarını içeren bir JSON dosyası olarak kaydedilmelidir.

---

### Requirement 4: Label Separability Analysis

**User Story:** Araştırmacı olarak, her dataset için label embedding'leri arasındaki semantik örtüşmeyi ölçmek istiyorum; böylece "dataset zorluğu semantik örtüşmeye bağlıdır" hipotezini test edebileyim.

#### Acceptance Criteria

1. THE `Label_Separability_Analyzer` SHALL her dataset için tüm label çiftleri arasındaki cosine similarity matrisini hesaplamalıdır.
2. THE `Label_Separability_Analyzer` SHALL off-diagonal cosine similarity değerlerinin ortalamasını `Separability_Score` olarak raporlamalıdır.
3. WHEN `Separability_Score` hesaplandığında, THE `Label_Separability_Analyzer` SHALL bu değeri her label modu (L1, L2, L3) için ayrı ayrı hesaplamalıdır.
4. THE `Label_Separability_Analyzer` SHALL sonuçları dataset adı, label modu ve `Separability_Score` sütunlarını içeren bir tablo olarak kaydetmelidir.
5. WHEN tüm dataset'ler için `Separability_Score` hesaplandığında, THE `Label_Separability_Analyzer` SHALL dataset'leri `Separability_Score`'a göre sıralayarak raporlamalıdır.
6. IF bir dataset için label sayısı 1'den fazla değilse, THEN THE `Label_Separability_Analyzer` SHALL bu dataset'i analiz dışında bırakmalı ve bir uyarı mesajı loglamalıdır.
7. THE `Label_Separability_Analyzer` SHALL `Separability_Score` ile ortalama Macro-F1 arasındaki Pearson korelasyon katsayısını hesaplamalıdır.

---

### Requirement 5: Task × Label Interaction Analysis

**User Story:** Araştırmacı olarak, farklı task type'larının (topic, sentiment, intent, emotion) label semantics'ten ne ölçüde faydalandığını görselleştirmek istiyorum; böylece hangi görev türlerinde L2/L3'ün L1'e göre anlamlı kazanım sağladığını gösterebileyim.

#### Acceptance Criteria

1. THE `Task_Interaction_Analyzer` SHALL her (task_type, label_mode) kombinasyonu için ortalama Macro-F1 değerini hesaplamalıdır.
2. THE `Task_Interaction_Analyzer` SHALL task_type × label_mode heatmap'ini üretmeli ve `reports/` dizinine kaydetmelidir.
3. THE `Task_Interaction_Analyzer` SHALL her dataset için ΔF1(L2-L1) ve ΔF1(L3-L1) değerlerini hesaplamalıdır.
4. THE `Task_Interaction_Analyzer` SHALL ΔF1 gain plot'larını üretmeli ve `reports/` dizinine kaydetmelidir.
5. WHEN task_type × label_mode analizi yapıldığında, THE `Task_Interaction_Analyzer` SHALL her dataset'e bir `Task_Type` etiketi atamalıdır: `ag_news` → `topic`, `dbpedia_14` → `topic`, `yahoo_answers_topics` → `topic`, `SetFit/20_newsgroups` → `topic`, `banking77` → `intent`, `go_emotions` → `emotion`, `imdb` → `sentiment`, `sst2` → `sentiment`, `zeroshot/twitter-financial-news-sentiment` → `sentiment`.
6. IF bir dataset için herhangi bir label moduna ait sonuç eksikse, THEN THE `Task_Interaction_Analyzer` SHALL eksik değeri `NaN` olarak işaretlemeli ve analizi kalan verilerle tamamlamalıdır.

---

### Requirement 6: Model Sensitivity Analysis

**User Story:** Araştırmacı olarak, her modelin label representation değişikliklerine ne kadar duyarlı olduğunu ölçmek istiyorum; böylece bazı modellerin label semantics'e daha robust olup olmadığını gösterebileyim.

#### Acceptance Criteria

1. THE `Model_Sensitivity_Analyzer` SHALL her model için L1, L2 ve L3 Macro-F1 değerlerinin varyansını hesaplamalıdır: `variance(F1_L1, F1_L2, F1_L3)`.
2. THE `Model_Sensitivity_Analyzer` SHALL varyans değerlerini kullanarak modelleri sensitivity'e göre sıralayan bir tablo üretmelidir.
3. WHEN model sensitivity tablosu üretildiğinde, THE `Model_Sensitivity_Analyzer` SHALL her model için ortalama varyans değerini tüm dataset'ler üzerinden raporlamalıdır.
4. THE `Model_Sensitivity_Analyzer` SHALL sonuçları `reports/` dizinine CSV ve görsel (bar chart) formatında kaydetmelidir.
5. IF bir model için herhangi bir label moduna ait sonuç eksikse, THEN THE `Model_Sensitivity_Analyzer` SHALL bu modeli eksik veri uyarısıyla birlikte analize dahil etmeli ve mevcut verilerle varyansı hesaplamalıdır.

---

### Requirement 7: Sonuç Raporlama ve Reproducibility

**User Story:** Araştırmacı olarak, tüm analizlerin tekrarlanabilir ve raporlanabilir olmasını istiyorum; böylece paper submission sürecinde sonuçları doğrulayabileyim.

#### Acceptance Criteria

1. THE `Runner` SHALL her experiment için random seed değerini (`seed: 42`) metadata'ya kaydetmelidir.
2. THE tüm analiz scriptleri, `results/` dizinindeki mevcut JSON dosyalarından okuyarak çalışmalıdır; experiment'ları yeniden çalıştırmamalıdır.
3. THE `reports/` dizini, her analiz türü için ayrı alt dizinler içermelidir: `reports/separability/`, `reports/task_interaction/`, `reports/model_sensitivity/`.
4. WHEN bir analiz scripti çalıştırıldığında, THE script SHALL hangi input dosyalarını kullandığını ve hangi output dosyalarını ürettiğini loglamalıdır.
5. THE tüm üretilen görseller, en az 300 DPI çözünürlükte PNG formatında kaydedilmelidir.
6. THE tüm üretilen tablolar, CSV formatında kaydedilmelidir; böylece bağımsız doğrulama mümkün olsun.
