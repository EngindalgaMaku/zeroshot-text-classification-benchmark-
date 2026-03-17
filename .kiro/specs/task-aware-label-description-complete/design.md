# Design Document: Task-Aware Label Description System

## Overview

This design addresses the completion and validation of a task-aware label description generation system for research publication. The system generates semantic descriptions for classification labels using LLM-based generation with task-specific prompt templates across 9 datasets and 5 task types.

The current implementation has three critical issues:
1. ValidationEngine incorrectly rejects valid descriptions due to overly strict label matching
2. Generic prefix list flags valid task-specific phrases (e.g., "the user wants" for intent tasks)
3. L3 validation compares L2 with concatenated L3 instead of individual sentences

The design focuses on fixing these bugs, implementing comprehensive test suites, and adding quality metrics for research reproducibility. The system must support L2 (single-sentence) and L3 (three-sentence) descriptions with proper validation, provenance tracking, and statistical analysis.

## Architecture

The system follows a pipeline architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  TemplateStore → TaskAwareLabelGenerator → ValidationEngine │
│       ↓                    ↓                       ↓         │
│  Prompt Templates    LLM API Calls         Quality Rules    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage & Metadata                        │
├─────────────────────────────────────────────────────────────┤
│  Generated Descriptions + Provenance + Generation Metadata  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Analysis & Testing                        │
├─────────────────────────────────────────────────────────────┤
│  Per-Dataset Tests │ Validation Tests │ Quality Metrics     │
│  Format Tests      │ Task Frame Tests │ Statistical Tests   │
└─────────────────────────────────────────────────────────────┘
```

Key architectural principles:
- Immutable prompt templates with version tracking
- Stateless validation engine for parallel processing
- Comprehensive provenance for reproducibility
- Separation of generation, validation, and analysis

## Components and Interfaces

### 1. ValidationEngine (Fixed)

The ValidationEngine validates generated descriptions against quality rules. The current implementation has three bugs that must be fixed:

**Current Issues:**
- Uses exact word boundary matching for label text (rejects valid substring matches)
- Includes "the user" in generic prefix list (rejects valid intent descriptions)
- Compares L2 with concatenated L3 instead of individual sentences

**Fixed Interface:**
```python
class ValidationEngine:
    def __init__(
        self,
        generic_prefixes: List[str],
        task_specific_allowed_prefixes: Dict[str, List[str]]
    ):
        """
        Initialize with configurable prefix lists.
        
        Args:
            generic_prefixes: Phrases that indicate low-quality descriptions
            task_specific_allowed_prefixes: Task-specific phrases that are valid
        """
        pass
    
    def validate_single(
        self, 
        label_text: str, 
        description: str,
        task_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single description.
        
        Rules:
        1. Description must contain label_text as substring (case-insensitive)
        2. For intent labels with underscores, accept both underscore and space versions
        3. Description must not start with generic prefixes (unless task-specific allowed)
        4. Description must be 12-25 words
        
        Returns:
            ValidationResult with pass/fail and detailed error message
        """
        pass
    
    def validate_l3_list(
        self,
        label_text: str,
        l2: str,
        l3_list: List[str],
        task_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate L3 descriptions against L2.
        
        Rules:
        1. Each L3 sentence must contain label_text
        2. Compare L2 with EACH individual L3 sentence (not concatenation)
        3. At least one L3 sentence must differ from L2
        4. All three L3 sentences must be distinct from each other
        
        Returns:
            ValidationResult with pass/fail and detailed error message
        """
        pass
```

**Key Changes:**
1. Label matching: Use case-insensitive substring search instead of word boundaries
2. Intent label handling: Accept both "activate_my_card" and "activate my card"
3. Generic prefix filtering: Add task_type parameter to allow task-specific exceptions
4. L3 validation: Compare L2 with each L3 sentence individually
5. Error messages: Include label_text, description snippet, and specific rule violated

### 2. PromptTemplate (Fixed)

The intent prompt templates must be updated to preserve underscores in labels.

**Fixed Templates:**
```yaml
intent:
  l2: 'You are generating an L2 label description for intent classification. 
       Write 1 sentence (12-25 words). You MUST include the exact label text 
       "{label_name}" verbatim in your description - use underscores as shown. 
       Describe the user goal/request. Dataset: {dataset_name}. Label: {label_name}. 
       Example: For label "transfer_funds", write: "The user wants to transfer_funds 
       between accounts securely."'
  
  l3: 'You are generating an L3 label description for intent classification. 
       Return EXACTLY 3 numbered sentences (12-25 words each). Each sentence MUST 
       include the exact label text "{label_name}" verbatim with underscores. 
       Cover different paraphrases or contexts. Dataset: {dataset_name}. 
       Label: {label_name}.'
```

**Key Changes:**
1. Explicit instruction to preserve underscores verbatim
2. Concrete example showing underscore preservation
3. Repeated emphasis on "exact label text" with underscores

### 3. TaskAwareLabelGenerator (Enhanced)

Enhanced with detailed logging and error handling.

**Interface:**
```python
class TaskAwareLabelGenerator:
    def __init__(
        self,
        template_store: TemplateStore,
        dataset_config: DatasetTaskTypeConfig,
        validation_engine: ValidationEngine,
        llm_client: Any,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize with dependencies."""
        pass
    
    def generate_for_label(
        self,
        dataset_id: str,
        label_text: str
    ) -> Tuple[str, List[str], GenerationRecord]:
        """
        Generate L2 and L3 descriptions for a single label.
        
        Returns:
            (l2_description, l3_list, generation_record)
        
        Raises:
            GenerationError: If LLM generation fails
            ValidationError: If generated descriptions fail validation
        """
        pass
    
    def generate_batch(
        self,
        dataset_id: str,
        labels: List[str],
        progress_interval: int = 10
    ) -> BatchGenerationResult:
        """
        Generate descriptions for multiple labels with progress logging.
        
        Logs progress every progress_interval labels.
        Returns summary statistics on completion.
        """
        pass
```

**Enhanced Logging:**
- Log dataset_id, label_text, task_type on generation start
- Log prompt template substitution values
- Log progress every 10 labels during batch generation
- Log summary statistics (success/failure counts, total time) on completion
- Include full error details in exception messages

### 4. Test Suites

Four independent test suite components:

**PerDatasetTestSuite:**
```python
class PerDatasetTestSuite:
    """Test generation for each of the 9 datasets independently."""
    
    def test_ag_news(self):
        """Generate and validate all ag_news labels."""
        pass
    
    def test_banking77(self):
        """Generate and validate all banking77 labels."""
        pass
    
    # ... one test method per dataset
```

**ValidationTestSuite:**
```python
class ValidationTestSuite:
    """Validate all generated descriptions meet quality standards."""
    
    def test_l2_contains_label(self):
        """Verify all L2 descriptions contain original label text."""
        pass
    
    def test_l3_contains_label(self):
        """Verify all L3 descriptions contain original label text."""
        pass
    
    def test_no_generic_prefixes(self):
        """Verify no descriptions start with generic prefixes."""
        pass
    
    def test_l2_l3_distinct(self):
        """Verify L2 and L3 descriptions are distinct."""
        pass
    
    def test_l3_sentences_distinct(self):
        """Verify all three L3 sentences are distinct."""
        pass
    
    def compute_validation_pass_rate(self) -> float:
        """Return percentage of descriptions passing all rules."""
        pass
```

**FormatTestSuite:**
```python
class FormatTestSuite:
    """Validate format requirements (length, structure)."""
    
    def test_l2_single_sentence(self):
        """Verify each L2 is exactly 1 sentence."""
        pass
    
    def test_l2_word_count(self):
        """Verify each L2 is 12-25 words."""
        pass
    
    def test_l3_three_sentences(self):
        """Verify each L3 has exactly 3 sentences."""
        pass
    
    def test_l3_sentence_word_count(self):
        """Verify each L3 sentence is 12-25 words."""
        pass
    
    def compute_format_compliance_rate(self) -> float:
        """Return percentage meeting format requirements."""
        pass
```

**TaskFrameTestSuite:**
```python
class TaskFrameTestSuite:
    """Validate task-appropriate semantic framing."""
    
    def test_topic_vocabulary(self):
        """Verify topic descriptions use topic-related vocabulary."""
        pass
    
    def test_entity_vocabulary(self):
        """Verify entity descriptions mention entity types."""
        pass
    
    def test_sentiment_vocabulary(self):
        """Verify sentiment descriptions mention polarity/tone."""
        pass
    
    def test_emotion_vocabulary(self):
        """Verify emotion descriptions mention emotional states."""
        pass
    
    def test_intent_vocabulary(self):
        """Verify intent descriptions mention user goals/actions."""
        pass
    
    def compute_task_frame_compliance(self) -> Dict[str, float]:
        """Return compliance rate per task type."""
        pass
```

### 5. Quality Metrics Components

**EmbeddingQualityAnalyzer:**
```python
class EmbeddingQualityAnalyzer:
    """Compute embedding similarity metrics."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model."""
        pass
    
    def compute_l2_l3_similarity(
        self,
        descriptions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Compute cosine similarity between L2 and each L3 sentence.
        
        Returns per-label and per-dataset statistics (mean, std).
        """
        pass
    
    def compute_intra_l3_similarity(
        self,
        descriptions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Compute pairwise similarity between L3 sentences.
        
        Returns per-label and per-dataset statistics.
        """
        pass
    
    def save_json(self, path: Path) -> None:
        """Save results in JSON format."""
        pass
```

**SemanticConsistencyChecker:**
```python
class SemanticConsistencyChecker:
    """Flag descriptions with consistency issues."""
    
    def check_consistency(
        self,
        similarity_results: Dict[str, Any]
    ) -> ConsistencyReport:
        """
        Flag labels based on similarity thresholds:
        - L2-L3 similarity < 0.5: potentially inconsistent
        - L2-L3 similarity > 0.95: potentially redundant
        - Intra-L3 similarity > 0.9: insufficiently diverse
        
        Returns report with flagged labels and counts per dataset.
        """
        pass
```

**LengthDistributionAnalyzer:**
```python
class LengthDistributionAnalyzer:
    """Analyze word count distributions."""
    
    def analyze(
        self,
        descriptions: Dict[str, Dict]
    ) -> LengthAnalysisReport:
        """
        Compute word count statistics per dataset:
        - Mean, median, min, max, std
        - Percentage within 12-25 word range
        - Histogram plots
        
        Returns report with statistics and flagged outliers.
        """
        pass
```

**VocabularyOverlapAnalyzer:**
```python
class VocabularyOverlapAnalyzer:
    """Measure lexical diversity between descriptions."""
    
    def analyze(
        self,
        descriptions: Dict[str, Dict]
    ) -> VocabularyAnalysisReport:
        """
        Compute Jaccard similarity between word sets:
        - L2 vs L3 word sets
        - Pairwise L3 sentence word sets
        - Mean overlap per dataset
        - Flag high overlap (>0.8) and low overlap (<0.2)
        
        Returns report with statistics and flagged labels.
        """
        pass
```

**ComparisonTool:**
```python
class ComparisonTool:
    """Compare manual vs automatic descriptions."""
    
    def compare(
        self,
        manual_descriptions: Dict[str, Dict],
        automatic_descriptions: Dict[str, Dict]
    ) -> ComparisonReport:
        """
        Compute embedding similarity between manual and automatic.
        
        Returns:
        - Mean similarity per dataset
        - Labels with significant differences (similarity < 0.6)
        - Side-by-side comparison CSV
        """
        pass
```

**PerformanceDeltaAnalyzer:**
```python
class PerformanceDeltaAnalyzer:
    """Measure impact on classification accuracy."""
    
    def analyze(
        self,
        manual_results: Dict[str, Any],
        automatic_results: Dict[str, Any]
    ) -> PerformanceDeltaReport:
        """
        Compute F1 score delta (automatic - manual).
        
        Returns:
        - Delta per dataset and embedding model
        - Mean delta and std across datasets
        - Datasets with significant degradation (delta < -0.02)
        - JSON output with per-dataset and per-model statistics
        """
        pass
```

**StatisticalSignificanceTester:**
```python
class StatisticalSignificanceTester:
    """Test statistical significance of performance differences."""
    
    def test(
        self,
        manual_scores: Dict[str, List[float]],
        automatic_scores: Dict[str, List[float]]
    ) -> StatisticalTestReport:
        """
        Perform paired t-tests with Bonferroni correction.
        
        Returns:
        - P-values per dataset
        - Datasets with significant differences (p < 0.05)
        - Effect sizes
        - Formatted table output
        """
        pass
```

### 6. Metadata and Provenance

**GenerationMetadata:**
```python
@dataclass
class GenerationMetadata:
    """Complete metadata for reproducibility."""
    model_name: str  # e.g., "gpt-4o-mini-2024-07-18"
    model_version: str
    timestamp: str  # ISO 8601 with timezone
    prompt_template_version: str
    temperature: float  # Must be 0 for reproducibility
    random_seed: Optional[int]
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        pass
```

**ProvenanceRecord:**
```python
@dataclass
class ProvenanceRecord:
    """Provenance for a single label description."""
    generation_id: str  # Unique ID
    dataset_name: str
    label_id: str
    label_text: str
    task_type: str
    prompt_template_l2: str
    prompt_template_l3: str
    l2_description: str
    l3_descriptions: List[str]
    generation_timestamp: str
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        pass
```

**ValidationReport:**
```python
class ValidationReport:
    """Track validation pass rates over time."""
    
    def compute_pass_rates(
        self,
        descriptions: Dict[str, Dict]
    ) -> ValidationPassRateReport:
        """
        Compute validation pass rates:
        - Overall pass rate per dataset
        - Pass rate per validation rule
        - Most common validation failures
        
        Returns report in JSON and human-readable text formats.
        """
        pass
```

**ReproducibilityDocGenerator:**
```python
class ReproducibilityDocGenerator:
    """Generate reproducibility documentation."""
    
    def generate(
        self,
        metadata: GenerationMetadata,
        validation_config: Dict[str, Any]
    ) -> str:
        """
        Generate markdown documentation with:
        - Python package versions
        - LLM API endpoint and model version
        - Configuration parameters
        - Prompt templates with versions
        - Validation rules and thresholds
        
        Returns markdown string to save in results directory.
        """
        pass
```

## Data Models

### Core Data Structures

**ValidationResult:**
```python
@dataclass
class ValidationResult:
    """Result of validation check."""
    passed: bool
    rule_name: str
    error_message: Optional[str]
    label_text: str
    description_snippet: str  # First 50 chars
```

**GenerationRecord:**
```python
@dataclass
class GenerationRecord:
    """Record of a single generation attempt."""
    dataset_id: str
    label_text: str
    task_type: str
    l2_description: str
    l3_descriptions: List[str]
    timestamp: str
    model_name: str
    temperature: float
    validation_passed: bool
    validation_errors: List[str]
```

**BatchGenerationResult:**
```python
@dataclass
class BatchGenerationResult:
    """Result of batch generation."""
    success_count: int
    failure_count: int
    total_time: float
    records: List[GenerationRecord]
    errors: List[Tuple[str, str]]  # (label_text, error_message)
```

### Analysis Data Structures

**ConsistencyReport:**
```python
@dataclass
class ConsistencyReport:
    """Semantic consistency analysis results."""
    inconsistent_labels: List[Tuple[str, float]]  # (label, similarity)
    redundant_labels: List[Tuple[str, float]]
    low_diversity_labels: List[Tuple[str, float]]
    flagged_count_per_dataset: Dict[str, int]
```

**LengthAnalysisReport:**
```python
@dataclass
class LengthAnalysisReport:
    """Word count distribution analysis."""
    per_dataset_stats: Dict[str, Dict[str, float]]  # mean, median, min, max, std
    compliance_rate: Dict[str, float]  # % within 12-25 words
    outliers: List[Tuple[str, int]]  # (label, word_count)
    histogram_paths: Dict[str, str]  # dataset -> plot path
```

**VocabularyAnalysisReport:**
```python
@dataclass
class VocabularyAnalysisReport:
    """Lexical diversity analysis."""
    mean_overlap_per_dataset: Dict[str, float]
    high_overlap_labels: List[Tuple[str, float]]  # >0.8
    low_overlap_labels: List[Tuple[str, float]]  # <0.2
```

**ComparisonReport:**
```python
@dataclass
class ComparisonReport:
    """Manual vs automatic comparison."""
    mean_similarity_per_dataset: Dict[str, float]
    significant_differences: List[Tuple[str, float]]  # similarity < 0.6
    csv_path: str
```

**PerformanceDeltaReport:**
```python
@dataclass
class PerformanceDeltaReport:
    """Classification performance impact."""
    delta_per_dataset: Dict[str, Dict[str, float]]  # dataset -> model -> delta
    mean_delta: float
    std_delta: float
    degraded_datasets: List[Tuple[str, float]]  # delta < -0.02
    json_path: str
```

**StatisticalTestReport:**
```python
@dataclass
class StatisticalTestReport:
    """Statistical significance test results."""
    p_values: Dict[str, float]  # dataset -> p-value
    significant_datasets: List[str]  # p < 0.05 after Bonferroni
    effect_sizes: Dict[str, float]
    formatted_table: str
```

### Storage Formats

**Generated Descriptions JSON:**
```json
{
  "dataset_name": {
    "label_text": {
      "l2": "Single sentence description",
      "l3": [
        "First sentence",
        "Second sentence",
        "Third sentence"
      ]
    }
  }
}
```

**Provenance JSON:**
```json
{
  "records": [
    {
      "generation_id": "uuid",
      "dataset_name": "banking77",
      "label_id": "0",
      "label_text": "activate_my_card",
      "task_type": "intent",
      "prompt_template_l2": "...",
      "prompt_template_l3": "...",
      "l2_description": "...",
      "l3_descriptions": ["...", "...", "..."],
      "generation_timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Generation Metadata JSON:**
```json
{
  "model_name": "gpt-4o-mini-2024-07-18",
  "model_version": "2024-07-18",
  "timestamp": "2024-01-15T10:30:00Z",
  "prompt_template_version": "v6",
  "temperature": 0,
  "random_seed": null
}
```

