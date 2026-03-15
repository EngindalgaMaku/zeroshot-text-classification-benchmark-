# Requirements Document: TACL Benchmark Strengthening

## Introduction

This document specifies requirements for strengthening a zero-shot text classification benchmark for submission to the TACL (Transactions of the Association for Computational Linguistics) journal. The benchmark currently compares 7 sentence embedding models across 7 datasets using Macro-F1 scores, with all 49 experiments completed. However, critical reproducibility issues and insufficient analysis depth have been identified that must be addressed before journal submission.

The system encompasses experimental infrastructure, data processing pipelines, analysis scripts, and visualization tools that together produce publication-quality research outputs.

## Glossary

- **Benchmark_System**: The complete experimental framework including code, configurations, data processing, and analysis tools
- **Experiment_Runner**: The main.py script and associated pipeline that executes zero-shot classification experiments
- **Config_File**: YAML files in experiments/ directory that specify dataset, model, and evaluation parameters
- **Results_Database**: CSV files in results/ directory containing experimental outcomes
- **Analysis_Pipeline**: Scripts in scripts/ directory that generate tables, figures, and statistical analyses
- **Reproducibility_Seed**: Random seed value (42) used to ensure deterministic behavior across runs
- **Label_Mode**: The format used for class labels (name_only vs description)
- **Batch_Size**: Number of samples processed simultaneously during encoding
- **Sample_Size**: Number of test examples evaluated per dataset
- **Split_Type**: Dataset partition used for evaluation (test vs validation)
- **TACL_Submission**: The complete research package including paper, code, and supplementary materials

## Requirements

### Requirement 1: Reproducibility Foundation

**User Story:** As a researcher, I want all experiments to be fully reproducible with fixed random seeds, so that reviewers can verify results and the community can build upon this work.

#### Acceptance Criteria

1. WHEN THE Experiment_Runner executes any experiment, THE Benchmark_System SHALL use Reproducibility_Seed value 42 for all random operations
2. THE Benchmark_System SHALL apply Reproducibility_Seed to dataset sampling, NumPy operations, PyTorch operations, and CUDA operations
3. WHEN all 49 experiments are re-executed with identical Config_Files, THE Benchmark_System SHALL produce identical Macro-F1 scores within 0.01 tolerance
4. THE Benchmark_System SHALL document the seed value in experiment output files and logs
5. THE Benchmark_System SHALL set CUDA deterministic mode when GPU is available

### Requirement 2: Experimental Consistency Validation

**User Story:** As a researcher, I want to identify and fix all inconsistencies in experimental configurations, so that model comparisons are fair and valid.

#### Acceptance Criteria

1. THE Benchmark_System SHALL use identical Sample_Size values across all datasets for the same model
2. THE Benchmark_System SHALL use identical Batch_Size values across all models except where hardware constraints require reduction
3. THE Benchmark_System SHALL use test Split_Type for all datasets except where test split is unavailable
4. THE Benchmark_System SHALL use identical Label_Mode (description) across all experiments
5. WHEN inconsistencies are detected in Config_Files, THE Benchmark_System SHALL generate a validation report listing all discrepancies

### Requirement 3: GoEmotions Multi-Label Resolution

**User Story:** As a researcher, I want to properly handle the GoEmotions multi-label dataset, so that the evaluation methodology is scientifically sound and clearly justified.

#### Acceptance Criteria

1. THE Benchmark_System SHALL document the multi-label to single-label conversion strategy for GoEmotions
2. THE Benchmark_System SHALL provide justification for taking the first emotion label as the dominant label
3. WHERE GoEmotions is retained, THE Benchmark_System SHALL include analysis of how multi-label simplification affects results
4. THE Benchmark_System SHALL support a configuration option to exclude GoEmotions from the benchmark
5. THE Benchmark_System SHALL analyze label co-occurrence patterns in GoEmotions to validate the single-label assumption

### Requirement 4: Batch Size Standardization

**User Story:** As a researcher, I want consistent batch sizes across models, so that performance differences reflect model quality rather than implementation details.

#### Acceptance Criteria

1. THE Benchmark_System SHALL use Batch_Size 16 as the standard for all models (chosen to accommodate Qwen memory constraints)
2. THE Benchmark_System SHALL document that batch size 16 was selected to prevent memory overflow on Qwen models while maintaining consistency
3. THE Benchmark_System SHALL verify that batch size variations do not affect Macro-F1 scores beyond 0.5 percentage points
4. THE Benchmark_System SHALL include batch size in all Config_Files explicitly
5. THE Config_Files SHALL include a comment explaining that batch_size 16 is used for consistency across all models

### Requirement 5: Sample Size Standardization

**User Story:** As a researcher, I want consistent sample sizes across datasets, so that statistical comparisons are valid and fair.

#### Acceptance Criteria

1. THE Benchmark_System SHALL use Sample_Size 1000 as the standard for all datasets
2. WHERE a dataset requires different Sample_Size, THE Benchmark_System SHALL document the rationale in the Config_File
3. THE Benchmark_System SHALL generate a sample size report showing actual vs configured values for all experiments
4. THE Benchmark_System SHALL validate that all experiments use the configured Sample_Size
5. WHEN 20 Newsgroups uses Sample_Size 2000, THE Config_File SHALL include justification for the larger sample

### Requirement 6: Complete Experiment Re-execution

**User Story:** As a researcher, I want to re-run all 49 experiments with corrected configurations, so that published results are based on properly controlled experiments.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL execute all 49 model-dataset combinations with standardized configurations
2. THE Experiment_Runner SHALL save results with timestamps to distinguish new runs from old runs
3. THE Experiment_Runner SHALL generate a comparison report showing differences between old and new results
4. THE Experiment_Runner SHALL complete all 49 experiments within 48 hours on available hardware
5. WHEN any experiment fails, THE Experiment_Runner SHALL log the error and continue with remaining experiments

### Requirement 7: Label Formulation Analysis

**User Story:** As a researcher, I want to compare name_only vs description label modes, so that I can demonstrate the impact of label formulation on zero-shot performance.

#### Acceptance Criteria

1. THE Benchmark_System SHALL support both name_only and description Label_Mode options in Config_Files
2. THE Benchmark_System SHALL execute label formulation experiments on at least 3 diverse datasets
3. THE Analysis_Pipeline SHALL generate a comparison table showing Macro-F1 differences between label modes
4. THE Analysis_Pipeline SHALL produce a visualization showing label mode impact across datasets
5. THE Benchmark_System SHALL analyze which task types benefit most from description labels

### Requirement 8: Task Characteristics Analysis

**User Story:** As a researcher, I want to analyze relationships between task characteristics and model performance, so that I can provide insights into when different models excel.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL compute correlation between number of classes and Macro-F1 scores
2. THE Analysis_Pipeline SHALL analyze correlation between average text length and model performance
3. THE Analysis_Pipeline SHALL compute label semantic similarity scores for each dataset
4. THE Analysis_Pipeline SHALL generate scatter plots showing task characteristics vs performance
5. THE Analysis_Pipeline SHALL identify which task characteristics most strongly predict model performance

### Requirement 9: Model Stability Analysis

**User Story:** As a researcher, I want to quantify model stability across datasets, so that I can recommend models based on both performance and robustness.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL compute coefficient of variation for each model across all datasets
2. THE Analysis_Pipeline SHALL generate a stability ranking table ordering models by consistency
3. THE Analysis_Pipeline SHALL produce a scatter plot showing mean performance vs stability for each model
4. THE Analysis_Pipeline SHALL identify models with best stability-performance trade-offs
5. THE Analysis_Pipeline SHALL analyze whether high-performing models are also stable

### Requirement 10: Error Analysis and Confusion Matrices

**User Story:** As a researcher, I want to analyze error patterns and confusion matrices, so that I can explain why certain tasks are difficult and where models fail.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL generate confusion matrices for at least 3 representative datasets
2. THE Analysis_Pipeline SHALL identify the top 5 most confused class pairs for each dataset
3. THE Analysis_Pipeline SHALL analyze error patterns specific to GoEmotions (fine-grained emotions)
4. THE Analysis_Pipeline SHALL analyze error patterns specific to Yahoo Answers (broad categories)
5. THE Analysis_Pipeline SHALL produce visualizations showing common error patterns across models

### Requirement 11: Dataset Expansion Evaluation

**User Story:** As a researcher, I want to evaluate whether adding more datasets strengthens the benchmark, so that I can make informed decisions about benchmark scope.

#### Acceptance Criteria

1. THE Benchmark_System SHALL support adding new datasets through Config_File creation
2. THE Benchmark_System SHALL provide a decision framework for dataset inclusion based on diversity and coverage
3. THE Benchmark_System SHALL analyze whether current 7 datasets provide sufficient task-type coverage
4. WHERE new datasets are added, THE Benchmark_System SHALL ensure they use standardized configurations
5. THE Benchmark_System SHALL document the rationale for including or excluding each candidate dataset

### Requirement 12: Model Selection Evaluation

**User Story:** As a researcher, I want to evaluate whether the current model selection is optimal, so that the benchmark represents the state-of-the-art fairly.

#### Acceptance Criteria

1. THE Benchmark_System SHALL document the rationale for including each of the 7 models
2. THE Benchmark_System SHALL analyze whether Snowflake model should be retained given poor performance
3. THE Benchmark_System SHALL evaluate candidate models (GTE, UAE) for potential inclusion
4. THE Benchmark_System SHALL ensure model selection represents diverse architectures and training approaches
5. THE Benchmark_System SHALL document model selection criteria in supplementary materials

### Requirement 13: Publication-Quality Visualizations

**User Story:** As a researcher, I want all figures to meet TACL publication standards, so that the paper is visually professional and clear.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL generate all figures in vector format (PDF or EPS)
2. THE Analysis_Pipeline SHALL use consistent color schemes, fonts, and styling across all figures
3. THE Analysis_Pipeline SHALL ensure all figure text is readable at publication scale
4. THE Analysis_Pipeline SHALL generate at least 6 publication-quality figures for the paper
5. THE Analysis_Pipeline SHALL include figure captions and labels that are self-explanatory

### Requirement 14: Statistical Significance Testing

**User Story:** As a researcher, I want rigorous statistical tests comparing models, so that performance claims are scientifically valid.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL perform Friedman test to detect overall performance differences
2. THE Analysis_Pipeline SHALL perform post-hoc Nemenyi test for pairwise model comparisons
3. THE Analysis_Pipeline SHALL generate critical difference diagrams showing statistical groupings
4. THE Analysis_Pipeline SHALL report p-values and effect sizes for all statistical tests
5. THE Analysis_Pipeline SHALL validate that sample sizes are sufficient for statistical power

### Requirement 15: Reproducibility Documentation

**User Story:** As a researcher, I want comprehensive reproducibility documentation, so that other researchers can replicate and extend this work.

#### Acceptance Criteria

1. THE Benchmark_System SHALL document all software dependencies with exact version numbers
2. THE Benchmark_System SHALL provide step-by-step instructions for running all experiments
3. THE Benchmark_System SHALL document hardware requirements and expected runtime
4. THE Benchmark_System SHALL include all Config_Files in the code repository
5. THE Benchmark_System SHALL provide scripts to regenerate all figures and tables from raw results

### Requirement 16: Results Validation and Archiving

**User Story:** As a researcher, I want to validate new results against old results and archive both versions, so that I can track changes and ensure correctness.

#### Acceptance Criteria

1. THE Benchmark_System SHALL archive old results before re-running experiments
2. THE Benchmark_System SHALL generate a validation report comparing old vs new Macro-F1 scores
3. WHEN differences exceed 2 percentage points, THE Benchmark_System SHALL flag the experiment for manual review
4. THE Benchmark_System SHALL maintain a changelog documenting all configuration changes
5. THE Benchmark_System SHALL preserve both old and new results in separate directories

### Requirement 17: Twitter Financial Split Consistency

**User Story:** As a researcher, I want to document why Twitter Financial uses validation split, so that this methodological choice is transparent and justified.

#### Acceptance Criteria

1. THE Benchmark_System SHALL verify that Twitter Financial dataset lacks a test split
2. THE Benchmark_System SHALL document in Config_File comments why validation split is used
3. THE Benchmark_System SHALL analyze whether validation split affects comparability with other datasets
4. THE Benchmark_System SHALL include this methodological note in paper supplementary materials
5. WHERE alternative Twitter Financial datasets exist with test splits, THE Benchmark_System SHALL evaluate switching datasets

### Requirement 18: Comprehensive Results Database

**User Story:** As a researcher, I want a clean, well-structured results database, so that analysis scripts can easily access and process experimental outcomes.

#### Acceptance Criteria

1. THE Results_Database SHALL store all experiments in a single CSV file with consistent schema
2. THE Results_Database SHALL include columns for dataset, model, macro_f1, accuracy, weighted_f1, samples, and experiment_name
3. THE Results_Database SHALL include metadata columns for batch_size, label_mode, and split_type
4. THE Results_Database SHALL validate that all 49 experiments have entries before analysis
5. THE Results_Database SHALL support filtering and aggregation operations for analysis scripts

### Requirement 19: Analysis Script Modularity

**User Story:** As a researcher, I want modular analysis scripts, so that I can easily add new analyses and regenerate specific outputs.

#### Acceptance Criteria

1. THE Analysis_Pipeline SHALL separate data loading, computation, and visualization into distinct functions
2. THE Analysis_Pipeline SHALL support command-line arguments for selecting specific analyses
3. THE Analysis_Pipeline SHALL cache intermediate computations to speed up iterative analysis
4. THE Analysis_Pipeline SHALL provide a master script that regenerates all outputs in correct order
5. THE Analysis_Pipeline SHALL log progress and timing information for each analysis step

### Requirement 20: TACL Submission Package

**User Story:** As a researcher, I want a complete TACL submission package, so that I can submit with confidence that all materials are included and properly formatted.

#### Acceptance Criteria

1. THE TACL_Submission SHALL include the main paper in TACL LaTeX format
2. THE TACL_Submission SHALL include supplementary materials with full experimental details
3. THE TACL_Submission SHALL include a code repository with all scripts and configurations
4. THE TACL_Submission SHALL include a README with setup and execution instructions
5. THE TACL_Submission SHALL include all raw results and analysis outputs for reviewer verification
