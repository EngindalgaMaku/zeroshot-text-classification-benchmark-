#!/usr/bin/env python3
"""
Validation script for split types and label modes across experiment configurations.

This script:
1. Parses all YAML files in experiments/ directory
2. Checks that all non-Twitter Financial configs use split: test
3. Checks that Twitter Financial configs use split: validation
4. Verifies all configs have label_mode: description
5. Generates a validation report

Requirements: 2.3, 2.4, 17.1, 17.2, 17.3
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def load_yaml_config(filepath: Path) -> Dict:
    """Load and parse a YAML configuration file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def is_twitter_financial_config(config: Dict, filename: str) -> bool:
    """Check if a config is for Twitter Financial dataset."""
    dataset_name = config.get('dataset', {}).get('name', '')
    return 'twitter-financial' in dataset_name.lower() or 'twitter_financial' in filename.lower()


def validate_config(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Validate a single config file for split and label_mode consistency.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        config = load_yaml_config(filepath)
        filename = filepath.name
        is_twitter = is_twitter_financial_config(config, filename)
        
        # Check split type
        split = config.get('dataset', {}).get('split')
        if is_twitter:
            if split != 'validation':
                issues.append(f"Twitter Financial config should use split: validation, found: {split}")
        else:
            if split != 'test':
                issues.append(f"Non-Twitter Financial config should use split: test, found: {split}")
        
        # Check label_mode
        label_mode = config.get('task', {}).get('label_mode')
        if label_mode != 'description':
            issues.append(f"Config should use label_mode: description, found: {label_mode}")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        issues.append(f"Error parsing config: {str(e)}")
        return False, issues


def main():
    """Main validation function."""
    experiments_dir = Path('experiments')
    
    # Find all YAML files
    yaml_files = list(experiments_dir.glob('*.yaml')) + list(experiments_dir.glob('*.yml'))
    yaml_files = sorted(yaml_files)
    
    print("=" * 80)
    print("Split Type and Label Mode Validation Report")
    print("=" * 80)
    print()
    
    all_valid = True
    twitter_configs = []
    non_twitter_configs = []
    invalid_configs = []
    
    for filepath in yaml_files:
        is_valid, issues = validate_config(filepath)
        
        config = load_yaml_config(filepath)
        is_twitter = is_twitter_financial_config(config, filepath.name)
        
        if is_twitter:
            twitter_configs.append(filepath.name)
        else:
            non_twitter_configs.append(filepath.name)
        
        if not is_valid:
            all_valid = False
            invalid_configs.append((filepath.name, issues))
    
    # Summary statistics
    print(f"Total configs analyzed: {len(yaml_files)}")
    print(f"Twitter Financial configs: {len(twitter_configs)}")
    print(f"Non-Twitter Financial configs: {len(non_twitter_configs)}")
    print()
    
    # List Twitter Financial configs
    print("Twitter Financial Configs (should use split: validation):")
    print("-" * 80)
    for name in twitter_configs:
        print(f"  - {name}")
    print()
    
    # Report issues
    if invalid_configs:
        print("VALIDATION ISSUES FOUND:")
        print("-" * 80)
        for filename, issues in invalid_configs:
            print(f"\n{filename}:")
            for issue in issues:
                print(f"  - {issue}")
        print()
    else:
        print("✓ All configurations are valid!")
        print()
    
    # Detailed validation results
    print("Detailed Validation Results:")
    print("-" * 80)
    for filepath in yaml_files:
        config = load_yaml_config(filepath)
        is_twitter = is_twitter_financial_config(config, filepath.name)
        split = config.get('dataset', {}).get('split', 'NOT SET')
        label_mode = config.get('task', {}).get('label_mode', 'NOT SET')
        
        is_invalid = any(f == filepath.name for f, _ in invalid_configs)
        status = "✗" if is_invalid else "✓"
        dataset_type = "Twitter Financial" if is_twitter else "Other"
        
        print(f"{status} {filepath.name}")
        print(f"   Dataset Type: {dataset_type}")
        print(f"   Split: {split}")
        print(f"   Label Mode: {label_mode}")
        print()
    
    print("=" * 80)
    if all_valid:
        print("VALIDATION PASSED: All configurations meet requirements")
    else:
        print("VALIDATION FAILED: Issues found in configurations")
    print("=" * 80)
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())
