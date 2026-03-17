# Requirements Document

## Introduction

This document specifies requirements for completing and testing the task-aware label description system. The system generates semantic descriptions for classification labels using LLM-based generation with task-specific prompt templates. The current implementation has validation bugs, incomplete testing, and lacks quality metrics needed for paper publication.

The system must support 9 datasets across 5 task types (topic, entity, sentiment, emotion, intent), generate L2 (single-sentence) and L3 (three-sentence) descriptions, validate all outputs, and provide comprehensive quality metrics for research reproducibility.

## Glossary

- **ValidationEngine**: Component that validates generated descriptions against quality rules
- **PromptTemplate**: Task-specific template for LLM generation with placeholders for dataset and label
- **L2_Description**: Single-sentence (12-25 words) semantic description of a label
- **L3_Description**: Three-sentence description with different aspects of the same label
- **TaskFrame**: Semantic framing appropriate for task type (e.g., "user wants" for intent, "text is about" for topic)
- **LabelText**: Original label name from dataset (e.g., "activate_my_card", "World")
- **GenericPrefix**: Overly generic phrase that indicates low-quality description
- **EmbeddingQuality**: Semantic consistency measured via cosine similarity in embedding space
- **Provenance**: Complete record of generation metadata for reproducibility
- **PerDatasetTest**: Independent test suite that validates one dataset in isolation
- **RoundTripProperty**: Property that parse(format(x)) == x for serialization validation


## Requirements

### Requirement 1: Fix ValidationEngine Label Matching

**User Story:** As a developer, I want the ValidationEngine to correctly validate descriptions, so that valid descriptions are not incorrectly rejected.

#### Acceptance Criteria

1. WHEN a description contains the label text as a substring (case-insensitive), THE ValidationEngine SHALL accept the description
2. WHEN validating intent labels with underscores (e.g., "activate_my_card"), THE ValidationEngine SHALL accept descriptions containing space-separated versions (e.g., "activate my card")
3. WHEN validating intent labels with underscores, THE ValidationEngine SHALL accept descriptions containing the exact underscore version
4. THE ValidationEngine SHALL NOT require exact word boundary matching for label text
5. WHEN a description fails validation, THE ValidationEngine SHALL return a descriptive error message indicating which rule failed

### Requirement 2: Fix Generic Prefix List

**User Story:** As a developer, I want the generic prefix list to be task-aware, so that valid task-specific phrases are not incorrectly flagged as generic.

#### Acceptance Criteria

1. THE ValidationEngine SHALL NOT include "the user" in the generic prefix list
2. WHEN validating intent task descriptions, THE ValidationEngine SHALL accept descriptions starting with "the user wants" or "the user intends"
3. THE ValidationEngine SHALL flag descriptions starting with "this text is about" as generic
4. THE ValidationEngine SHALL flag descriptions starting with "this question is about" as generic
5. THE ValidationEngine SHALL flag descriptions starting with "this text describes" as generic
6. THE ValidationEngine SHALL provide a method to configure task-specific allowed prefixes


### Requirement 3: Fix L3 Validation Logic

**User Story:** As a developer, I want L3 validation to correctly compare L2 with L3 descriptions, so that distinct multi-aspect descriptions are properly validated.

#### Acceptance Criteria

1. WHEN validating L3 descriptions, THE ValidationEngine SHALL compare L2 with each individual L3 sentence
2. THE ValidationEngine SHALL NOT compare L2 with the joined concatenation of all L3 sentences
3. WHEN all three L3 sentences are identical to L2, THE ValidationEngine SHALL reject the L3 list
4. WHEN at least one L3 sentence differs from L2, THE ValidationEngine SHALL accept the L3 list
5. THE ValidationEngine SHALL verify that all three L3 sentences are distinct from each other

### Requirement 4: Fix Intent Prompt Templates

**User Story:** As a developer, I want intent prompts to handle space-separated labels correctly, so that banking77 dataset generates successfully.

#### Acceptance Criteria

1. WHEN generating descriptions for intent labels with underscores, THE PromptTemplate SHALL instruct the LLM to include the exact label text with underscores
2. THE PromptTemplate SHALL provide an example showing underscore preservation (e.g., "transfer_funds")
3. WHEN generating L2 descriptions for intent labels, THE PromptTemplate SHALL specify that underscores should be preserved verbatim
4. WHEN generating L3 descriptions for intent labels, THE PromptTemplate SHALL specify that each sentence must include the exact label text with underscores
5. THE PromptTemplate SHALL NOT suggest replacing underscores with spaces in the generated description


### Requirement 5: Enhanced Error Messages and Logging

**User Story:** As a developer, I want detailed error messages and logging, so that I can quickly diagnose generation failures.

#### Acceptance Criteria

1. WHEN a validation error occurs, THE ValidationEngine SHALL include the label text, description snippet, and specific rule violated in the error message
2. WHEN LLM generation fails, THE TaskAwareLabelGenerator SHALL log the dataset name, label text, task type, and error details
3. WHEN a prompt template substitution fails, THE TaskAwareLabelGenerator SHALL log the template string and substitution values
4. THE TaskAwareLabelGenerator SHALL log progress every 10 labels during batch generation
5. WHEN generation completes, THE TaskAwareLabelGenerator SHALL log summary statistics (success count, failure count, total time)

### Requirement 6: Per-Dataset Test Suite

**User Story:** As a researcher, I want independent test suites for each dataset, so that I can verify generation works correctly for all 9 datasets.

#### Acceptance Criteria

1. THE PerDatasetTest SHALL test each of the 9 datasets independently (ag_news, yahoo_answers_topics, SetFit/20_newsgroups, dbpedia_14, banking77, imdb, sst2, zeroshot/twitter-financial-news-sentiment, go_emotions)
2. WHEN running per-dataset tests, THE PerDatasetTest SHALL generate descriptions for all labels in the dataset
3. WHEN generation succeeds, THE PerDatasetTest SHALL verify that all generated descriptions pass validation
4. WHEN generation fails for any label, THE PerDatasetTest SHALL report the specific label and error
5. THE PerDatasetTest SHALL report pass/fail status and generation time for each dataset


### Requirement 7: Validation Test Suite

**User Story:** As a researcher, I want automated validation tests, so that I can ensure all generated descriptions meet quality standards.

#### Acceptance Criteria

1. THE ValidationTest SHALL verify that all L2 descriptions contain the original label text
2. THE ValidationTest SHALL verify that all L3 descriptions contain the original label text
3. THE ValidationTest SHALL verify that no descriptions start with generic prefixes
4. THE ValidationTest SHALL verify that L2 and L3 descriptions are distinct for each label
5. THE ValidationTest SHALL verify that all three L3 sentences are distinct from each other
6. THE ValidationTest SHALL report validation pass rate as a percentage
7. WHEN validation fails, THE ValidationTest SHALL output a detailed report listing all failing labels with reasons

### Requirement 8: Format Test Suite

**User Story:** As a researcher, I want format validation tests, so that I can ensure descriptions meet length and structure requirements.

#### Acceptance Criteria

1. THE FormatTest SHALL verify that each L2 description is exactly 1 sentence
2. THE FormatTest SHALL verify that each L2 description contains 12-25 words
3. THE FormatTest SHALL verify that each L3 description contains exactly 3 sentences
4. THE FormatTest SHALL verify that each L3 sentence contains 12-25 words
5. THE FormatTest SHALL report format compliance rate as a percentage
6. WHEN format violations occur, THE FormatTest SHALL output a report listing labels with word counts and sentence counts


### Requirement 9: Task Frame Test Suite

**User Story:** As a researcher, I want task frame validation tests, so that I can verify descriptions use appropriate semantic framing for each task type.

#### Acceptance Criteria

1. WHEN testing topic task descriptions, THE TaskFrameTest SHALL verify descriptions use topic-related vocabulary (e.g., "about", "discusses", "covers")
2. WHEN testing entity task descriptions, THE TaskFrameTest SHALL verify descriptions define entity types or mention identification
3. WHEN testing sentiment task descriptions, THE TaskFrameTest SHALL verify descriptions mention polarity, tone, or evaluation
4. WHEN testing emotion task descriptions, THE TaskFrameTest SHALL verify descriptions mention emotional states or expressions
5. WHEN testing intent task descriptions, THE TaskFrameTest SHALL verify descriptions mention user goals, requests, or actions
6. THE TaskFrameTest SHALL report task frame compliance rate per task type

### Requirement 10: Embedding Similarity Metrics

**User Story:** As a researcher, I want embedding similarity metrics, so that I can measure semantic consistency of generated descriptions.

#### Acceptance Criteria

1. THE EmbeddingQuality SHALL compute cosine similarity between L2 and each L3 sentence for every label
2. THE EmbeddingQuality SHALL compute pairwise cosine similarity between all three L3 sentences for every label
3. THE EmbeddingQuality SHALL report mean and standard deviation of L2-L3 similarity per dataset
4. THE EmbeddingQuality SHALL report mean and standard deviation of intra-L3 similarity per dataset
5. THE EmbeddingQuality SHALL use the same embedding model as the classification experiments (e.g., all-MiniLM-L6-v2)
6. THE EmbeddingQuality SHALL output results in JSON format with per-label and per-dataset statistics


### Requirement 11: Semantic Consistency Checks

**User Story:** As a researcher, I want semantic consistency checks, so that I can identify descriptions that are too similar or too different from each other.

#### Acceptance Criteria

1. WHEN L2-L3 similarity is below 0.5, THE SemanticConsistency SHALL flag the label as potentially inconsistent
2. WHEN L2-L3 similarity is above 0.95, THE SemanticConsistency SHALL flag the label as potentially redundant
3. WHEN intra-L3 similarity is above 0.9, THE SemanticConsistency SHALL flag the L3 descriptions as insufficiently diverse
4. THE SemanticConsistency SHALL report the number of flagged labels per dataset
5. THE SemanticConsistency SHALL output a detailed report listing all flagged labels with similarity scores

### Requirement 12: Length Distribution Analysis

**User Story:** As a researcher, I want length distribution analysis, so that I can verify descriptions meet word count requirements consistently.

#### Acceptance Criteria

1. THE LengthAnalysis SHALL compute word count distribution for all L2 descriptions per dataset
2. THE LengthAnalysis SHALL compute word count distribution for all L3 sentences per dataset
3. THE LengthAnalysis SHALL report mean, median, min, max, and standard deviation of word counts
4. THE LengthAnalysis SHALL generate histogram plots showing word count distributions
5. THE LengthAnalysis SHALL flag descriptions outside the 12-25 word range
6. THE LengthAnalysis SHALL report the percentage of descriptions within the target range


### Requirement 13: Vocabulary Overlap Analysis

**User Story:** As a researcher, I want vocabulary overlap analysis, so that I can measure lexical diversity between L2 and L3 descriptions.

#### Acceptance Criteria

1. THE VocabularyAnalysis SHALL compute Jaccard similarity between L2 and L3 word sets for each label
2. THE VocabularyAnalysis SHALL compute pairwise Jaccard similarity between all three L3 sentences
3. THE VocabularyAnalysis SHALL report mean vocabulary overlap per dataset
4. THE VocabularyAnalysis SHALL identify labels with high vocabulary overlap (>0.8) as potentially redundant
5. THE VocabularyAnalysis SHALL identify labels with low vocabulary overlap (<0.2) as potentially inconsistent

### Requirement 14: Manual vs Automatic Comparison Tool

**User Story:** As a researcher, I want to compare manual and automatic descriptions, so that I can validate the quality of LLM-generated descriptions.

#### Acceptance Criteria

1. THE ComparisonTool SHALL load manual descriptions from the existing labels.py file
2. THE ComparisonTool SHALL load automatic descriptions from generated_descriptions.json
3. THE ComparisonTool SHALL compute embedding cosine similarity between manual and automatic descriptions for each label
4. THE ComparisonTool SHALL report mean similarity per dataset
5. THE ComparisonTool SHALL identify labels where manual and automatic descriptions differ significantly (similarity < 0.6)
6. THE ComparisonTool SHALL output a side-by-side comparison report in CSV format


### Requirement 15: Performance Delta Analysis

**User Story:** As a researcher, I want performance delta analysis, so that I can measure the impact of automatic descriptions on classification accuracy.

#### Acceptance Criteria

1. THE PerformanceDelta SHALL load experiment results for manual descriptions (L2 and L3)
2. THE PerformanceDelta SHALL load experiment results for automatic descriptions (L2 and L3)
3. THE PerformanceDelta SHALL compute F1 score delta (automatic - manual) for each dataset and embedding model
4. THE PerformanceDelta SHALL report mean delta and standard deviation across all datasets
5. THE PerformanceDelta SHALL identify datasets where automatic descriptions perform significantly worse (delta < -0.02)
6. THE PerformanceDelta SHALL output results in JSON format with per-dataset and per-model statistics

### Requirement 16: Statistical Significance Tests

**User Story:** As a researcher, I want statistical significance tests, so that I can determine if performance differences are meaningful.

#### Acceptance Criteria

1. THE StatisticalTest SHALL perform paired t-tests comparing manual vs automatic F1 scores per dataset
2. THE StatisticalTest SHALL compute p-values for each dataset comparison
3. THE StatisticalTest SHALL apply Bonferroni correction for multiple comparisons
4. THE StatisticalTest SHALL report which datasets show statistically significant differences (p < 0.05)
5. THE StatisticalTest SHALL output results in a formatted table with effect sizes


### Requirement 17: Generation Metadata with Model Version

**User Story:** As a researcher, I want complete generation metadata, so that I can reproduce results and track model versions.

#### Acceptance Criteria

1. THE GenerationMetadata SHALL record the exact model name and version used for generation (e.g., "gpt-4o-mini-2024-07-18")
2. THE GenerationMetadata SHALL record the timestamp of generation in ISO 8601 format with timezone
3. THE GenerationMetadata SHALL record the prompt template version used
4. THE GenerationMetadata SHALL record the temperature parameter (must be 0 for reproducibility)
5. THE GenerationMetadata SHALL record the random seed if applicable
6. THE GenerationMetadata SHALL be saved in JSON format alongside generated descriptions

### Requirement 18: Provenance Tracking

**User Story:** As a researcher, I want provenance tracking for all descriptions, so that I can trace each description back to its generation parameters.

#### Acceptance Criteria

1. THE Provenance SHALL record a unique generation ID for each label description
2. THE Provenance SHALL record the dataset name, label ID, and label text
3. THE Provenance SHALL record the task type and prompt templates used
4. THE Provenance SHALL record the exact L2 and L3 descriptions generated
5. THE Provenance SHALL record the generation timestamp
6. THE Provenance SHALL be saved in JSON format with one record per label

### Requirement 19: Validation Pass Rate Reporting

**User Story:** As a researcher, I want validation pass rate reporting, so that I can track the quality of generated descriptions over time.

#### Acceptance Criteria

1. THE ValidationReport SHALL compute the percentage of labels that pass all validation rules per dataset
2. THE ValidationReport SHALL compute the percentage of labels that pass each individual validation rule
3. THE ValidationReport SHALL report validation pass rates in a summary table
4. THE ValidationReport SHALL identify the most common validation failures
5. THE ValidationReport SHALL output results in both JSON and human-readable text formats

### Requirement 20: Reproducibility Documentation

**User Story:** As a researcher, I want reproducibility documentation, so that other researchers can replicate the generation process.

#### Acceptance Criteria

1. THE ReproducibilityDoc SHALL document the exact Python package versions used (openai, anthropic, etc.)
2. THE ReproducibilityDoc SHALL document the LLM API endpoint and model version
3. THE ReproducibilityDoc SHALL document all configuration parameters (temperature, max_tokens, etc.)
4. THE ReproducibilityDoc SHALL document the prompt templates with version numbers
5. THE ReproducibilityDoc SHALL document the validation rules and thresholds
6. THE ReproducibilityDoc SHALL be saved as a markdown file in the results directory

### Requirement 21: Banking77 Dataset Generation Success

**User Story:** As a researcher, I want banking77 dataset to generate successfully, so that I can include it in the paper results.

#### Acceptance Criteria

1. WHEN generating descriptions for banking77 dataset, THE TaskAwareLabelGenerator SHALL successfully generate L2 and L3 descriptions for all 77 labels
2. WHEN generating descriptions for banking77 labels with underscores, THE ValidationEngine SHALL accept descriptions that preserve underscores
3. WHEN generating descriptions for banking77 labels with underscores, THE ValidationEngine SHALL accept descriptions that use spaces instead of underscores
4. THE TaskAwareLabelGenerator SHALL complete banking77 generation without errors
5. THE TaskAwareLabelGenerator SHALL log successful generation of all 77 banking77 labels

### Requirement 22: Round-Trip Property for Description Serialization

**User Story:** As a developer, I want round-trip property testing for description serialization, so that I can ensure data integrity when saving and loading descriptions.

#### Acceptance Criteria

1. THE DescriptionSerializer SHALL serialize descriptions to JSON format
2. THE DescriptionParser SHALL parse descriptions from JSON format
3. FOR ALL valid description objects, THE system SHALL satisfy: parse(serialize(desc)) == desc
4. THE RoundTripTest SHALL verify this property for all generated descriptions
5. WHEN round-trip fails, THE RoundTripTest SHALL report the specific label and serialization error
