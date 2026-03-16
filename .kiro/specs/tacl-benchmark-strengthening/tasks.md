# Implementation Plan: TACL Benchmark Strengthening

## Overview

This plan addresses critical reproducibility issues and analysis gaps in a zero-shot text classification benchmark comparing 7 models across 7 datasets. The implementation focuses on: (1) fixing configuration inconsistencies, (2) re-running all 49 experiments with proper seeding, (3) implementing new analyses (label formulation, task characteristics, stability, error patterns), (4) generating publication-quality visualizations, and (5) preparing the complete TACL submission package.

## Tasks

- [ ] 1. Configuration validation and standardization
  - [x] 1.1 Create configuration validation script
    - Write Python script to parse all 49 YAML files in experiments/ directory
    - Check for inconsistencies in batch_size, max_samples, split, and label_mode
    - Generate validation report listing all discrepancies with file names and line numbers
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x] 1.2 Standardize batch sizes across configurations
    - Update all config files to use batch_size: 16 as standard (accommodates Qwen memory constraints)
    - Add explanatory comment in configs: "batch_size 16 used for consistency across all models"
    - Ensure batch_size is explicitly specified in all 49 YAML files
    - _Requirements: 4.1, 4.2, 4.4, 4.5_
  
  - [x] 1.3 Standardize sample sizes across configurations
    - Update all configs to use max_samples: 1000 except 20 Newsgroups (2000)
    - Add comment in 20 Newsgroups configs explaining larger sample size
    - Generate sample size report comparing configured vs actual values
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 1.4 Standardize split types and label modes
    - Ensure all configs use split: test except Twitter Financial (validation)
    - Add comment in Twitter Financial configs explaining validation split usage
    - Verify all configs use label_mode: description
    - _Requirements: 2.3, 2.4, 17.1, 17.2, 17.3_
  
  - [x] 1.5 Document GoEmotions multi-label handling
    - Add detailed comment block in GoEmotions configs explaining multi-label to single-label conversion
    - Document the "first emotion" strategy and its justification
    - Add configuration option to exclude GoEmotions from benchmark
    - _Requirements: 3.1, 3.2, 3.4_

- [ ] 2. Checkpoint - Validate all configurations
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Archive old results and prepare for re-execution
  - [x] 3.1 Create results archival script
    - Write Python script to move results/raw/*.json to results/archive/pre_seed_fix/
    - Preserve timestamps and create manifest file listing all archived results
    - Generate comparison baseline from archived results
    - _Requirements: 16.1, 16.5_
  
  - [x] 3.2 Verify seed implementation in main.py
    - Confirm set_seed(42) is called before config loading
    - Verify CUDA deterministic mode is enabled when GPU available
    - Add seed value logging to experiment output files
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [ ] 4. Execute all 49 experiments with corrected configurations
  - [ ] 4.1 Create batch experiment runner script
    - Write Python script to execute all 49 experiments sequentially
    - Add timestamp prefix to new result files to distinguish from old results
    - Implement error handling to continue on failure and log errors
    - Add progress tracking and estimated time remaining
    - _Requirements: 6.1, 6.2, 6.5_
  
  - [ ] 4.2 Run all experiments and monitor execution
    - Execute batch runner script for all 49 model-dataset combinations
    - Monitor for failures and document any issues
    - Verify all experiments complete within 48 hours
    - _Requirements: 6.1, 6.4_
  
  - [ ] 4.3 Generate old vs new results comparison report
    - Compare Macro-F1 scores between archived and new results
    - Flag experiments with differences exceeding 2 percentage points
    - Create visualization showing score changes across all experiments
    - Document configuration changes in changelog
    - _Requirements: 6.3, 16.2, 16.3, 16.4_

- [ ] 5. Checkpoint - Verify experiment completion
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement label formulation analysis
  - [x] 6.1 Create label mode comparison configs
    - Duplicate 3 diverse dataset configs (AG News, Banking77, GoEmotions)
    - Create name_only variants for each selected dataset
    - Ensure configs are identical except for label_mode parameter
    - _Requirements: 7.1, 7.2_
  
  - [x] 6.2 Run label formulation experiments
    - Execute experiments for both name_only and description modes
    - Collect results for comparison analysis
    - _Requirements: 7.2_
  
  - [x] 6.3 Generate label formulation analysis script
    - Write Python script to compare Macro-F1 between label modes
    - Create comparison table showing differences across datasets and models
    - Generate bar chart visualization showing label mode impact
    - Analyze which task types benefit most from descriptions
    - _Requirements: 7.3, 7.4, 7.5_

- [x] 7. Implement task characteristics analysis
  - [x] 7.1 Create task characteristics computation script
    - Compute number of classes for each dataset
    - Calculate average text length for each dataset
    - Compute label semantic similarity scores using sentence embeddings
    - Store characteristics in structured CSV file
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 7.2 Generate task characteristics correlation analysis
    - Correlate number of classes with model performance
    - Correlate text length with model performance
    - Correlate label similarity with model performance
    - Identify strongest predictors of performance
    - _Requirements: 8.1, 8.2, 8.5_
  
  - [x] 7.3 Create task characteristics visualizations
    - Generate scatter plots: num_classes vs Macro-F1
    - Generate scatter plots: text_length vs Macro-F1
    - Generate scatter plots: label_similarity vs Macro-F1
    - Use different colors for different models
    - _Requirements: 8.4_

- [x] 8. Implement model stability analysis
  - [x] 8.1 Create stability computation script
    - Calculate coefficient of variation for each model across datasets
    - Compute mean and standard deviation of Macro-F1 per model
    - Generate stability ranking table
    - _Requirements: 9.1, 9.2_
  
  - [x] 8.2 Generate stability visualizations
    - Create scatter plot: mean performance vs stability (CV)
    - Identify and annotate models with best stability-performance trade-offs
    - Add quadrant lines to show high/low performance and stability regions
    - _Requirements: 9.3, 9.4, 9.5_

- [x] 9. Implement error analysis and confusion matrices
  - [x] 9.1 Create confusion matrix generation script
    - Load prediction files for 3 representative datasets (AG News, Banking77, GoEmotions)
    - Generate confusion matrices for each model-dataset combination
    - Save matrices as publication-quality heatmaps
    - _Requirements: 10.1_
  
  - [x] 9.2 Implement error pattern analysis
    - Identify top 5 most confused class pairs for each dataset
    - Analyze GoEmotions fine-grained emotion confusions
    - Analyze Yahoo Answers broad category confusions
    - Generate summary tables of common error patterns
    - _Requirements: 10.2, 10.3, 10.4_
  
  - [x] 9.3 Create error pattern visualizations
    - Generate bar charts showing most confused class pairs
    - Create error pattern comparison across models
    - Visualize error patterns specific to each dataset type
    - _Requirements: 10.5_

- [ ] 10. Checkpoint - Review analysis outputs
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Evaluate and implement dataset expansion
  - [x] 11.1 Analyze current dataset coverage
    - Review task-type distribution across 7 current datasets
    - Identify gaps in task coverage (sentiment, topic, entity, intent, emotion)
    - Evaluate whether current datasets provide sufficient diversity
    - _Requirements: 11.3_
  
  - [x] 11.2 Evaluate candidate datasets for inclusion
    - Evaluate SST-2 (sentiment, binary classification)
    - Evaluate TREC (question classification, 6 classes)
    - Evaluate IMDB (sentiment, binary, longer texts)
    - Assess each candidate for: task diversity, dataset quality, public availability
    - _Requirements: 11.2, 11.5_
  
  - [x] 11.3 Create decision framework for dataset inclusion
    - Document criteria: task-type coverage, class balance, text length diversity
    - Document criteria: dataset size, quality, community adoption
    - Apply framework to candidate datasets and make inclusion decisions
    - _Requirements: 11.2, 11.5_
  
  - [x] 11.4 Implement selected new datasets
    - Add label definitions to src/labels.py for selected datasets
    - Create experiment configs for new datasets (all 7 models)
    - Update data loading in src/data.py if needed
    - _Requirements: 11.1, 11.4_
  
  - [x] 11.5 Run experiments for new datasets
    - Execute all model experiments for newly added datasets
    - Verify results are consistent with existing datasets
    - Update results database with new experiments
    - _Requirements: 11.1, 11.4_

- [x] 12. Evaluate and implement model expansion
  - [x] 12.1 Analyze current model selection
    - Document rationale for each of the 7 current models
    - Review Snowflake performance (consistently lowest) - consider removal
    - Identify architecture gaps in current model set
    - _Requirements: 12.1, 12.2, 12.4_
  
  - [x] 12.2 Evaluate candidate models for inclusion
    - Research recent high-performing models from MTEB leaderboard
    - Evaluate technical feasibility (API availability, model size, inference speed)
    - Test 1-2 promising candidates on sample dataset before full commitment
    - Document any technical issues encountered during testing
    - _Requirements: 12.3, 12.4_
  
  - [x] 12.3 Make model selection decisions
    - Analyze Snowflake performance: if consistently worst, consider removal for cleaner comparison
    - If removing Snowflake, ensure remaining models still represent architecture diversity
    - For new models: only add if technically feasible and adds architectural diversity
    - Document all model selection decisions with justification (performance, feasibility, diversity)
    - _Requirements: 12.2, 12.4, 12.5_
  
  - [x] 12.4 Implement selected new models (if any)
    - Add new model encoders to src/encoders.py (only if feasible candidates found)
    - Create experiment configs for new models (all datasets)
    - Test thoroughly on 2-3 datasets before committing to full benchmark
    - If technical issues arise, document and skip that model
    - _Requirements: 12.4_
  
  - [x] 12.5 Run experiments for new models (if any added)
    - Execute all dataset experiments for newly added models
    - Verify results are reasonable and comparable
    - Update results database with new model experiments
    - If no new models added, document rationale (technical constraints, current set sufficient)
    - _Requirements: 12.4_

- [ ] 13. Checkpoint - Verify expanded benchmark
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Enhance publication-quality visualizations
  - [x] 14.1 Update critical difference diagram script
    - Ensure vector format output (PDF and EPS)
    - Apply consistent publication styling (Times New Roman, 11pt)
    - Verify figure readability at publication scale
    - Add self-explanatory caption and labels
    - _Requirements: 13.1, 13.2, 13.3, 13.5_
  
  - [x] 14.2 Create comprehensive heatmap visualization
    - Generate model × dataset Macro-F1 heatmap
    - Use publication-quality color scheme and fonts
    - Add row/column averages and annotations
    - Export in PDF and EPS formats
    - _Requirements: 13.1, 13.2, 13.3, 13.5_
  
  - [x] 14.3 Create task type analysis visualization
    - Generate grouped bar chart comparing models across task types
    - Use consistent colors and styling with other figures
    - Ensure all text is readable at publication scale
    - Export in vector formats
    - _Requirements: 13.1, 13.2, 13.3, 13.5_
  
  - [x] 14.4 Create label formulation comparison figure
    - Generate side-by-side comparison of name_only vs description
    - Show performance differences across datasets
    - Use publication-quality styling
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [x] 14.5 Create stability-performance scatter plot
    - Generate scatter plot with model annotations
    - Add quadrant lines and trade-off curve
    - Use publication-quality styling
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [x] 14.6 Create error pattern visualization
    - Generate confusion matrix heatmaps for key datasets
    - Create error pattern comparison figure
    - Use publication-quality styling
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 15. Enhance statistical analysis
  - [x] 15.1 Verify Friedman test implementation
    - Confirm test correctly handles 7 models × 7 datasets
    - Verify p-value calculation and interpretation
    - Add effect size computation
    - _Requirements: 14.1, 14.4_
  
  - [x] 15.2 Verify Nemenyi post-hoc test implementation
    - Confirm critical distance calculation is correct
    - Verify pairwise comparison logic
    - Ensure clique detection works properly
    - _Requirements: 14.2, 14.4_
  
  - [x] 15.3 Add statistical power analysis
    - Compute statistical power given sample sizes
    - Verify sample sizes are sufficient for reliable conclusions
    - Document power analysis in supplementary materials
    - _Requirements: 14.5_

- [x] 16. Create reproducibility documentation
  - [x] 16.1 Create comprehensive README
    - Document all software dependencies with exact versions
    - Provide step-by-step setup instructions
    - Document hardware requirements and expected runtime
    - Include troubleshooting section
    - _Requirements: 15.1, 15.2, 15.3_
  
  - [x] 16.2 Create experiment execution guide
    - Document how to run individual experiments
    - Document how to run batch experiments
    - Document how to regenerate all analyses
    - Include example commands and expected outputs
    - _Requirements: 15.2, 15.5_
  
  - [x] 16.3 Create analysis regeneration script
    - Write master script that regenerates all figures and tables
    - Ensure script runs in correct dependency order
    - Add progress logging and timing information
    - _Requirements: 15.5, 19.4_
  
  - [x] 16.4 Document methodological choices
    - Document GoEmotions multi-label handling rationale
    - Document Twitter Financial validation split usage
    - Document batch size choices and constraints
    - Document sample size choices
    - _Requirements: 3.2, 17.4_

- [x] 17. Prepare TACL submission package
  - [x] 17.1 Organize code repository structure
    - Ensure all code is properly organized in src/, scripts/, experiments/
    - Remove unnecessary files and clean up repository
    - Verify all 49 config files are included
    - _Requirements: 20.3, 15.4_
  
  - [ ] 17.2 Create supplementary materials document
    - Include full experimental details and methodological notes
    - Include all configuration files and parameters
    - Include statistical test details and power analysis
    - Include dataset descriptions and preprocessing steps
    - _Requirements: 20.2, 17.4_
  
  - [ ] 17.3 Prepare results package
    - Include all raw results (JSON files)
    - Include all generated tables (CSV files)
    - Include all generated figures (PDF and EPS)
    - Include analysis outputs and logs
    - _Requirements: 20.5_
  
  - [ ] 17.4 Create submission README
    - Provide overview of submission contents
    - Include setup and execution instructions
    - Document how to reproduce all results
    - Include contact information and links
    - _Requirements: 20.4_

- [ ] 18. Final checkpoint - Verify submission completeness
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All experiments must use seed=42 for reproducibility
- Configuration standardization must be completed before re-running experiments
- Analysis scripts should be modular and reusable
- All figures must be in vector format (PDF/EPS) for publication
- Statistical tests are already implemented but need verification
- GoEmotions multi-label handling requires careful documentation
- Twitter Financial uses validation split due to missing test split
- All models use batch_size=16 for consistency (chosen to prevent Qwen memory overflow)
- 20 Newsgroups uses max_samples=2000 (larger than standard 1000)
