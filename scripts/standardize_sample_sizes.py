#!/usr/bin/env python3
"""
Script to standardize max_samples across all experiment configuration files.
Updates all YAML files in experiments/ directory to use max_samples: 1000,
except for 20 Newsgroups datasets which use max_samples: 2000.
Generates a CSV report showing configured vs actual sample sizes.
"""

import os
import csv
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_yaml_with_comments(filepath: Path) -> tuple[Dict[str, Any], str]:
    """Load YAML file and preserve original content for comment handling."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        data = yaml.safe_load(content)
    return data, content


def is_20newsgroups_config(data: Dict[str, Any]) -> bool:
    """Check if this is a 20 Newsgroups dataset configuration."""
    dataset_name = data.get('dataset', {}).get('name', '')
    return '20_newsgroups' in dataset_name.lower() or '20newsgroups' in dataset_name.lower()


def update_max_samples(data: Dict[str, Any]) -> tuple[Dict[str, Any], int, str]:
    """
    Update max_samples based on dataset type.
    Returns: (updated_data, target_samples, status)
    """
    if 'dataset' not in data:
        data['dataset'] = {}
    
    current_samples = data['dataset'].get('max_samples', None)
    is_20ng = is_20newsgroups_config(data)
    
    if is_20ng:
        target_samples = 2000
        if current_samples == target_samples:
            status = 'correct'
        else:
            status = 'updated'
            data['dataset']['max_samples'] = target_samples
    else:
        target_samples = 1000
        if current_samples == target_samples:
            status = 'correct'
        else:
            status = 'updated'
            data['dataset']['max_samples'] = target_samples
    
    return data, target_samples, status


def save_yaml_with_comment(filepath: Path, data: Dict[str, Any], is_20ng: bool):
    """Save YAML file with max_samples comment for 20 Newsgroups configs."""
    # Convert data to YAML
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Add comment before max_samples line for 20 Newsgroups
    lines = yaml_content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'max_samples:' in line and is_20ng:
            # Check if we're in the dataset section
            in_dataset_section = False
            for j in range(i-1, -1, -1):
                if 'dataset:' in lines[j]:
                    in_dataset_section = True
                    break
                if lines[j] and not lines[j].startswith(' '):
                    break
            
            if in_dataset_section:
                # Add comment before max_samples
                indent = len(line) - len(line.lstrip())
                comment = ' ' * indent + '# Larger sample size for 20 Newsgroups due to higher class count (20 classes)'
                new_lines.append(comment)
        new_lines.append(line)
    
    final_content = '\n'.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)


def generate_report(report_data: List[Dict[str, Any]], output_path: Path):
    """Generate CSV report of sample size standardization."""
    fieldnames = ['filename', 'dataset_name', 'configured_max_samples', 'status']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_data)
    
    print(f"\nReport saved to: {output_path}")


def main():
    """Main function to process all YAML files."""
    experiments_dir = Path('experiments')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Get all YAML files
    yaml_files = [
        f for f in experiments_dir.glob('*.yaml')
        if f.is_file()
    ]
    
    print(f"Found {len(yaml_files)} YAML files to process")
    
    updated_count = 0
    correct_count = 0
    report_data = []
    
    for yaml_file in sorted(yaml_files):
        try:
            print(f"Processing: {yaml_file.name}")
            
            # Load YAML
            data, original_content = load_yaml_with_comments(yaml_file)
            
            # Get dataset name
            dataset_name = data.get('dataset', {}).get('name', 'unknown')
            
            # Update max_samples
            data, target_samples, status = update_max_samples(data)
            
            # Check if it's 20 Newsgroups
            is_20ng = is_20newsgroups_config(data)
            
            # Save with comment if needed
            save_yaml_with_comment(yaml_file, data, is_20ng)
            
            # Track statistics
            if status == 'updated':
                updated_count += 1
                print(f"  ✓ Updated max_samples to {target_samples}")
            else:
                correct_count += 1
                print(f"  ✓ Already correct (max_samples: {target_samples})")
            
            # Add to report
            report_data.append({
                'filename': yaml_file.name,
                'dataset_name': dataset_name,
                'configured_max_samples': target_samples,
                'status': status
            })
            
        except Exception as e:
            print(f"  ✗ Error processing {yaml_file.name}: {e}")
            report_data.append({
                'filename': yaml_file.name,
                'dataset_name': 'error',
                'configured_max_samples': 'N/A',
                'status': 'error'
            })
    
    print(f"\nCompleted: {updated_count} files updated, {correct_count} files already correct")
    
    # Generate report
    report_path = results_dir / 'sample_size_standardization_report.csv'
    generate_report(report_data, report_path)
    
    # Print summary
    print("\nSummary:")
    print(f"  Total files: {len(yaml_files)}")
    print(f"  Updated: {updated_count}")
    print(f"  Already correct: {correct_count}")
    print(f"  20 Newsgroups configs (2000 samples): {sum(1 for r in report_data if r['configured_max_samples'] == 2000)}")
    print(f"  Other configs (1000 samples): {sum(1 for r in report_data if r['configured_max_samples'] == 1000)}")


if __name__ == '__main__':
    main()
