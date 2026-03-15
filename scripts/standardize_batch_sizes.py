#!/usr/bin/env python3
"""
Script to standardize batch_size across all experiment configuration files.
Updates all YAML files in experiments/ directory to use batch_size: 16.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml_with_comments(filepath: Path) -> tuple[Dict[str, Any], str]:
    """Load YAML file and preserve original content for comment handling."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        data = yaml.safe_load(content)
    return data, content


def update_batch_size(data: Dict[str, Any]) -> Dict[str, Any]:
    """Update or add batch_size to 16 in the pipeline section."""
    if 'pipeline' not in data:
        data['pipeline'] = {}
    
    data['pipeline']['batch_size'] = 16
    return data


def save_yaml_with_comment(filepath: Path, data: Dict[str, Any], original_content: str):
    """Save YAML file with batch_size comment."""
    # Convert data to YAML
    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Add comment before batch_size line
    lines = yaml_content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'batch_size:' in line and 'pipeline:' in '\n'.join(lines[:i]):
            # Add comment before batch_size
            indent = len(line) - len(line.lstrip())
            comment = ' ' * indent + '# batch_size 16 used for consistency across all models'
            new_lines.append(comment)
        new_lines.append(line)
    
    final_content = '\n'.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)


def main():
    """Main function to process all YAML files."""
    experiments_dir = Path('experiments')
    
    # Get all YAML files, excluding reranker subdirectory
    yaml_files = [
        f for f in experiments_dir.glob('*.yaml')
        if f.is_file()
    ]
    
    print(f"Found {len(yaml_files)} YAML files to process")
    
    updated_count = 0
    for yaml_file in sorted(yaml_files):
        try:
            print(f"Processing: {yaml_file.name}")
            
            # Load YAML
            data, original_content = load_yaml_with_comments(yaml_file)
            
            # Update batch_size
            data = update_batch_size(data)
            
            # Save with comment
            save_yaml_with_comment(yaml_file, data, original_content)
            
            updated_count += 1
            print(f"  ✓ Updated batch_size to 16")
            
        except Exception as e:
            print(f"  ✗ Error processing {yaml_file.name}: {e}")
    
    print(f"\nCompleted: {updated_count}/{len(yaml_files)} files updated")


if __name__ == '__main__':
    main()
