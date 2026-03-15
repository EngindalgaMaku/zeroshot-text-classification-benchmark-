# Dataset Inclusion Decision Framework

## Overview

This framework provides systematic criteria for evaluating whether to include new datasets in the TACL zero-shot text classification benchmark. The framework balances diversity, quality, and practical considerations to ensure the benchmark remains focused yet comprehensive.

## Evaluation Criteria

### 1. Task-Type Coverage (Weight: 25%)

**Objective:** Ensure diverse representation of text classification task types.

**Scoring:**
- **10 points:** Fills a critical gap (task type not represented)
- **7-9 points:** Adds diversity to underrepresented task type (only 1 dataset)
- **4-6 points:** Adds to moderately represented task type (2 datasets)
- **1-3 points:** Adds to well-represented task type (3+ datasets)

**Task Types:**
- Sentiment classification
- Topic classification
- Entity classification
- Intent classification
- Emotion classification
- Question classification

**Current Coverage (7 datasets):**
- Topic: 3 datasets (STRONG)
- Sentiment: 1 dataset (WEAK)
- Entity: 1 dataset (ADEQUATE)
- Intent: 1 dataset (ADEQUATE)
- Emotion: 1 dataset (ADEQUATE)

### 2. Domain Diversity (Weight: 20%)

**Objective:** Ensure benchmark covers diverse text domains and styles.

**Scoring:**
- **10 points:** Completely new domain
- **7-9 points:** Related but distinct domain
- **4-6 points:** Similar domain with different characteristics
- **1-3 points:** Overlapping domain with existing dataset

**Current Domains:**
- News (AG News)
- Knowledge base (DBPedia)
- Q&A forums (Yahoo Answers)
- Banking (Banking77)
- Financial news (Twitter Financial)
- Discussion forums (20 Newsgroups)
- Social media (GoEmotions)

### 3. Class Count Diversity (Weight: 15%)

**Objective:** Ensure representation across different classification granularities.

**Scoring:**
- **10 points:** Fills gap in class count distribution
- **7-9 points:** Adds to underrepresented range
- **4-6 points:** Adds to moderately represented range
- **1-3 points:** Adds to well-represented range

**Class Count Ranges:**
- Binary (2 classes)
- Few-class (3-5 classes)
- Medium (6-15 classes)
- Many-class (16-30 classes)
- Fine-grained (30+ classes)

**Current Distribution:**
- Binary: 0 datasets
- Few-class: 2 datasets (Twitter Financial: 3, AG News: 4)
- Medium: 3 datasets (Yahoo: 10, DBPedia: 14, 20 Newsgroups: 20)
- Many-class: 1 dataset (GoEmotions: 28)
- Fine-grained: 1 dataset (Banking77: 77)

### 4. Text Length Diversity (Weight: 15%)

**Objective:** Ensure representation of different text lengths.

**Scoring:**
- **10 points:** Fills critical gap in text length
- **7-9 points:** Adds to underrepresented length category
- **4-6 points:** Adds to moderately represented category
- **1-3 points:** Adds to well-represented category

**Length Categories:**
- Short: <50 tokens
- Medium: 50-150 tokens
- Long: 150+ tokens

**Current Distribution:**
- Short: 4 datasets (AG News, Banking77, Twitter Financial, GoEmotions)
- Medium: 2 datasets (DBPedia, Yahoo Answers)
- Long: 1 dataset (20 Newsgroups)

### 5. Dataset Quality (Weight: 15%)

**Objective:** Ensure high-quality, reliable annotations.

**Scoring:**
- **10 points:** Expert annotations, high inter-annotator agreement
- **7-9 points:** Crowdsourced with quality control
- **4-6 points:** Automated with manual validation
- **1-3 points:** Automated without validation

**Quality Indicators:**
- Annotation methodology documented
- Inter-annotator agreement reported
- Quality control measures described
- Known issues documented

### 6. Community Adoption (Weight: 10%)

**Objective:** Prioritize well-established benchmarks for comparability.

**Scoring:**
- **10 points:** Standard benchmark (GLUE, SuperGLUE, etc.)
- **7-9 points:** Widely used in research (100+ citations)
- **4-6 points:** Moderately used (10-100 citations)
- **1-3 points:** Rarely used (<10 citations)


## Decision Process

### Step 1: Calculate Weighted Score

For each candidate dataset:

```
Total Score = (Task Coverage × 0.25) + 
              (Domain Diversity × 0.20) + 
              (Class Diversity × 0.15) + 
              (Text Length × 0.15) + 
              (Quality × 0.15) + 
              (Adoption × 0.10)
```

Maximum possible score: 10.0

### Step 2: Apply Thresholds

- **8.0-10.0:** Strongly recommended for inclusion
- **6.0-7.9:** Recommended for inclusion
- **4.0-5.9:** Consider if specific gap needs addressing
- **<4.0:** Not recommended

### Step 3: Consider Practical Constraints

Even high-scoring datasets may be excluded if:
- Benchmark scope becomes too large (>12 datasets)
- Evaluation time becomes prohibitive
- Dataset has technical issues (API limits, licensing)
- Overlap with existing datasets is too high

### Step 4: Document Decision

For each evaluated dataset, document:
- Weighted score and breakdown
- Diversity contribution analysis
- Inclusion/exclusion decision
- Rationale for decision

## Application to Current Candidates

### IMDB Movie Reviews

**Scores:**
- Task Coverage: 9/10 (adds sentiment diversity)
- Domain Diversity: 9/10 (new domain: movie reviews)
- Class Diversity: 9/10 (adds binary classification)
- Text Length: 10/10 (long texts, fills critical gap)
- Quality: 10/10 (high quality, balanced)
- Adoption: 10/10 (standard benchmark)

**Weighted Score:** 9.4/10

**Decision:** ✅ **STRONGLY RECOMMENDED**

**Rationale:** Fills two critical gaps (sentiment diversity + long text). Highest impact per dataset added.

### SST-2 (Stanford Sentiment Treebank)

**Scores:**
- Task Coverage: 9/10 (adds sentiment diversity)
- Domain Diversity: 9/10 (new domain: movie reviews)
- Class Diversity: 9/10 (adds binary classification)
- Text Length: 3/10 (short texts, well-represented)
- Quality: 10/10 (GLUE benchmark, high quality)
- Adoption: 10/10 (standard benchmark)

**Weighted Score:** 8.3/10

**Decision:** ✅ **RECOMMENDED** (if adding 2 datasets)

**Rationale:** Strong addition but lower priority than IMDB. Best added alongside IMDB for comprehensive sentiment coverage.


### TREC Question Classification

**Scores:**
- Task Coverage: 3/10 (overlaps with Yahoo Answers)
- Domain Diversity: 3/10 (questions already represented)
- Class Diversity: 4/10 (6 classes, similar to existing)
- Text Length: 2/10 (very short, well-represented)
- Quality: 9/10 (classic benchmark)
- Adoption: 8/10 (well-studied)

**Weighted Score:** 4.3/10

**Decision:** ❌ **NOT RECOMMENDED**

**Rationale:** Insufficient diversity contribution. Overlaps significantly with Yahoo Answers.

## Final Recommendations

### Recommended Action: Add IMDB Only

**Justification:**
1. Highest weighted score (9.4/10)
2. Fills two critical gaps simultaneously
3. Minimal scope increase (7 → 8 datasets)
4. Clear, defensible inclusion rationale
5. Manageable experiment increase (49 → 56 experiments)

**Impact:**
- Sentiment datasets: 1 → 2
- Binary classification: 1 → 2
- Long text datasets: 1 → 2
- Total datasets: 7 → 8

### Alternative: Add IMDB + SST-2

**Justification:**
1. Comprehensive sentiment coverage
2. Both score above 8.0 threshold
3. Complementary (different text lengths)
4. Strong benchmark prestige (GLUE + standard)

**Impact:**
- Sentiment datasets: 1 → 3
- Binary classification: 1 → 3
- Total datasets: 7 → 9
- Total experiments: 49 → 63

**Trade-off:** More comprehensive but larger scope increase.

### Rationale for Excluding TREC

1. Score below 5.0 threshold (4.3/10)
2. High overlap with Yahoo Answers (both Q&A)
3. Minimal diversity contribution
4. Small test set (500) limits statistical power
5. No critical gap addressed

## Implementation Guidelines

If datasets are added, ensure:

1. **Standardized Configuration:**
   - batch_size: 16
   - max_samples: 1000
   - split: test
   - label_mode: description

2. **Label Definitions:**
   - Add to `src/labels.py`
   - Both name_only and description modes
   - Clear, descriptive labels

3. **Experiment Configs:**
   - Create configs for all 7 models
   - Follow naming convention: `exp_{dataset}_{model}.yaml`
   - Include explanatory comments

4. **Documentation:**
   - Update README with new datasets
   - Document inclusion rationale
   - Update dataset statistics

## Conclusion

The decision framework provides a systematic, defensible approach to dataset inclusion. For the TACL submission:

- **Minimum viable:** Current 7 datasets (adequate coverage)
- **Recommended:** Add IMDB (addresses critical gaps)
- **Comprehensive:** Add IMDB + SST-2 (thorough sentiment coverage)

The framework can be reused for future dataset evaluations.
