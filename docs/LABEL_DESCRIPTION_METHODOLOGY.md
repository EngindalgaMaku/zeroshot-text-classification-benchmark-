# Label Description Methodology

## 📝 Zero-Shot Classification için Label Tanımlama Metodolojisi

Bu dokümanda, zero-shot text classification için kullanılan label description'ların nasıl oluşturulduğunu açıklıyoruz.

---

## 🎯 Genel İlkeler

### 1. Label Mode Türleri

Üç farklı label mode kullanıyoruz:

1. **name_only**: Sadece sınıf ismi (örn: "sports", "business")
2. **description**: Tek açıklayıcı cümle
3. **multi_description**: Çoklu paraphrase'ler (3 farklı ifade)

---

## ✍️ Description Yazma Kuralları

### Genel Format

Tüm description'lar aşağıdaki template'i izler:

```
"This text is about [topic], [keywords], or [related concepts]."
```

**İngilizce için:**
```python
"This text is about [topic], [sub-topics], [keywords], or [related terms]."
```

**Türkçe için:**
```python
"Bu metin [konu], [alt konular], [anahtar kelimeler] veya [ilgili kavramlar] hakkındadır."
```

---

## 📋 Dataset-Specific Kurallar

### AG News (4 class)

**Kaynak:** Orijinal AG News dataset class isimleri
**Yaklaşım:** Newswire domain için spesifik keyword'ler

```python
"name_only": {
    0: ["world"],
    1: ["sports"],
    2: ["business"],
    3: ["science and technology"]
}

"description": {
    0: ["This text is about international events, global politics, diplomacy, conflicts, or world affairs."],
    1: ["This text is about sports, matches, teams, athletes, tournaments, or competitions."],
    2: ["This text is about business, markets, finance, companies, trade, or the economy."],
    3: ["This text is about science, technology, computers, innovation, research, or digital products."]
}
```

**Kurallar:**
- Her description 10-15 keyword içerir
- Domain-specific terimler (örn: "diplomacy", "tournaments")
- Virgülle ayrılmış, "or" ile bağlanmış liste formatı

---

### DBpedia-14 (14 class)

**Kaynak:** DBpedia ontology class isimleri
**Yaklaşım:** Entity-focused descriptions

```python
"description": {
    0: ["This text describes a company, corporation, or business organization."],
    1: ["This text describes an educational institution, school, university, or academy."],
    ...
}
```

**Kurallar:**
- "This text **describes** a..." formatı (entity için)
- Synonym'ler dahil (örn: "company, corporation")
- 3-5 alternative term

---

### Yahoo Answers (10 class)

**Kaynak:** Yahoo Answers topic taxonomy
**Yaklaşım:** Question-focused descriptions

```python
"description": {
    0: ["This question is about society, culture, social issues, traditions, or cultural practices."],
    1: ["This question is about science, mathematics, physics, chemistry, biology, or scientific concepts."],
    ...
}
```

**Kurallar:**
- "This **question** is about..." formatı (Q&A için)
- Geniş kapsam (örn: tüm scientific disciplines)
- Topic hierarchies gözetilir

---

### Banking77 (77 class)

**Kaynak:** Banking intent taxonomy
**Yaklaşım:** User intent descriptions

```python
"description": {
    0: ["The user wants to activate their card or asking how to activate it."],
    21: ["The user wants to change their PIN code."],
    ...
}
```

**Kurallar:**
- "The **user** wants/needs/is asking..." formatı
- Action-oriented (user intent)
- Specific use case description

---

### Turkish Datasets

**Kaynak:** Türkçe dataset class isimleri
**Yaklaşım:** Turkish grammar ve natural expression

```python
"description": {
    0: ["Bu metin siyaset, politika, hükümet veya siyasi haberler kategorisindedir."],
    1: ["Bu metin dünya haberleri, uluslararası gelişmeler veya dış politika kategorisindedir."],
    ...
}
```

**Kurallar:**
- "Bu metin [konu] hakkındadır/kategorisindedir" formatı
- Türkçe natural language flow
- Keyword'ler domain'e uygun (örn: "siyaset, politika, hükümet")

---

## 🔬 Metodolojik Kararlar

### 1. Uzunluk

**Tercih:** 10-15 kelime (description mode için)

**Neden:**
- Çok kısa → Belirsiz (örn: sadece "sports")
- Çok uzun → Noise, irrelevant bilgi
- 10-15 kelime → Optimal semantic richness

### 2. Keyword Seçimi

**Strateji:** 
1. Class name'den başla (örn: "sports")
2. Direct synonyms ekle (örn: "athletics")
3. Sub-topics ekle (örn: "matches, teams, athletes")
4. Related terms ekle (örn: "tournaments, competitions")

**Örnek:**
```
sports → athletics → matches, teams, athletes → tournaments, competitions
```

### 3. Template Consistency

**Aynı dataset içinde:**
- Aynı template kullan (örn: hep "This text is about...")
- Aynı yapı (örn: hep virgülle liste)
- Aynı uzunluk seviyesi

**Farklı dataset'ler arası:**
- Domain'e uygun template (örn: entity için "describes", intent için "user wants")

---

## 📊 Multi-Description Mode

`multi_description` mode için 3 farklı paraphrase:

**Kural:**
1. İlk: Genel overview
2. İkinci: "The main topic is..." formatı
3. Üçüncü: "This article discusses..." formatı

**Örnek (AG News - World):**
```python
"multi_description": {
    0: [
        "This text is about international news, diplomacy, wars, or world affairs.",  # Genel
        "The main topic of this text is global politics or international events.",    # Main topic
        "This article discusses world news, foreign policy, or global conflicts.",    # Article discusses
    ]
}
```

---

## 🎓 Makalede Nasıl Belirteceğiz?

### Method Section'da:

```markdown
## Label Representations

We define three label modes for zero-shot classification:

1. **Name-only**: Simple class names (e.g., "sports", "business")

2. **Description**: Single descriptive sentence per class following 
   a consistent template format:
   - For news: "This text is about [topic], [keywords]..."
   - For entities: "This text describes a [entity type]..."
   - For intents: "The user wants to [action]..."

3. **Multi-description**: Three paraphrased descriptions per class
   to test robustness to label formulation.

All descriptions are manually crafted following these principles:
- Consistent template within each dataset
- 10-15 keywords per description
- Domain-appropriate vocabulary
- Natural language formulation

For Turkish datasets, descriptions are translated and adapted
to maintain natural Turkish grammar and expression.
```

### Appendix'te:

Full label definitions for reproducibility → `src/labels.py` kodunu ekle

---

## ✅ Reproducibility

**Tam şeffaflık için:**
1. Bu döküman → Methodology açıklaması
2. `src/labels.py` → Tüm label definitions (kod)
3. Paper appendix → Full label list

**Böylece:**
- Herkes aynı label'ları kullanabilir
- Reproducible research
- Fair comparison

---

## 🔍 Önemli Notlar

### Bias Kontrolü

✅ **Yaptığımız:**
- Her class için eşit detay seviyesi
- Balanced keyword sayısı
- Neutral language (no positive/negative bias)

❌ **Yapmadığımız:**
- Specific brand/entity isimleri (örn: "Google", "Microsoft")
- Temporal references (örn: "recent", "modern")
- Opinionated adjectives (örn: "best", "worst")

### Domain Expertise

- **AG News, DBpedia, Yahoo:** Public taxonomy kullanıldı
- **Banking77:** Original paper'daki intent descriptions baz alındı
- **Turkish:** Domain uzmanı (native speaker) tarafından yazıldı

---

## 📝 Örnek: Tam Süreç (AG News - Sports)

1. **Class name:** "sports" (dataset'ten)
2. **Synonyms:** athletics, games
3. **Sub-topics:** matches, teams, athletes
4. **Related terms:** tournaments, competitions
5. **Template:** "This text is about..."
6. **Final:** "This text is about sports, matches, teams, athletes, tournaments, or competitions."

**Quality check:**
- ✅ Clear and unambiguous
- ✅ 10-15 keywords
- ✅ Domain-appropriate
- ✅ Natural language
- ✅ Consistent with other classes

---

## 🎯 Sonuç

**Standardımız:**
- ✅ Tutarlı template kullanımı
- ✅ 10-15 keyword optimal uzunluk
- ✅ Domain-specific terminology
- ✅ Natural language formulation
- ✅ Balanced coverage across classes
- ✅ Reproducible ve transparent

**Makale için:**
- Method section'da açıkla
- Appendix'te full list ver
- GitHub'da kod paylaş