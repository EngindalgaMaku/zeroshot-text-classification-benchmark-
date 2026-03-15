"""
Archive old results before re-running experiments with corrected configurations.

This script:
1. Moves all JSON files from results/raw/ to results/archive/pre_seed_fix/
2. Preserves timestamps and file metadata
3. Creates a manifest file listing all archived results
4. Generates a comparison baseline from archived results

Requirements: 16.1, 16.5
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd


def create_archive_directory():
    """Create archive directory structure."""
    archive_dir = Path("results/archive/pre_seed_fix")
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def get_json_files(source_dir):
    """Get all JSON files from source directory."""
    source_path = Path(source_dir)
    json_files = list(source_path.glob("*.json"))
    return json_files


def archive_files(json_files, archive_dir):
    """
    Archive JSON files to the archive directory.
    
    Args:
        json_files: List of Path objects for JSON files
        archive_dir: Path to archive directory
        
    Returns:
        List of dictionaries with file metadata
    """
    archived_files = []
    
    for json_file in json_files:
        # Get file metadata
        stat = json_file.stat()
        file_info = {
            "original_path": str(json_file),
            "filename": json_file.name,
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "archived_time": datetime.now().isoformat()
        }
        
        # Copy file to archive (preserve metadata)
        dest_path = archive_dir / json_file.name
        shutil.copy2(json_file, dest_path)
        file_info["archive_path"] = str(dest_path)
        
        archived_files.append(file_info)
        print(f"✓ Archived: {json_file.name}")
    
    return archived_files


def create_manifest(archived_files, archive_dir):
    """Create manifest file listing all archived results."""
    manifest_path = archive_dir / "archive_manifest.json"
    
    manifest = {
        "archive_date": datetime.now().isoformat(),
        "total_files": len(archived_files),
        "files": archived_files
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Created manifest: {manifest_path}")
    return manifest_path


def generate_comparison_baseline(json_files, archive_dir):
    """
    Generate comparison baseline from archived results.
    
    Extracts key metrics from all archived JSON files and creates
    a baseline CSV for comparison with new results.
    """
    baseline_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract experiment name from filename (remove _metrics.json suffix)
            experiment_name = json_file.stem.replace("_metrics", "")
            
            # Extract key metrics
            record = {
                "experiment_name": experiment_name,
                "macro_f1": data.get("macro_f1", None),
                "accuracy": data.get("accuracy", None),
                "weighted_f1": data.get("weighted_f1", None),
                "num_samples": data.get("num_samples", None),
                "num_classes": data.get("num_classes", None),
                "dataset": data.get("dataset", None),
                "model": data.get("model", None)
            }
            
            baseline_data.append(record)
            
        except Exception as e:
            print(f"⚠ Warning: Could not process {json_file.name}: {e}")
    
    # Create baseline DataFrame
    baseline_df = pd.DataFrame(baseline_data)
    
    # Save baseline CSV
    baseline_path = archive_dir / "baseline_results.csv"
    baseline_df.to_csv(baseline_path, index=False)
    
    print(f"✓ Created baseline: {baseline_path}")
    print(f"  - Total experiments: {len(baseline_df)}")
    print(f"  - Mean Macro-F1: {baseline_df['macro_f1'].mean():.4f}")
    print(f"  - Std Macro-F1: {baseline_df['macro_f1'].std():.4f}")
    
    return baseline_path


def delete_original_files(json_files):
    """Delete original JSON files after successful archival."""
    for json_file in json_files:
        json_file.unlink()
        print(f"✓ Deleted original: {json_file.name}")


def main():
    """Main archival process."""
    print("="*70)
    print("Results Archival Script")
    print("="*70)
    print()
    
    # Create archive directory
    print("📁 Creating archive directory...")
    archive_dir = create_archive_directory()
    print(f"✓ Archive directory: {archive_dir}")
    print()
    
    # Get JSON files
    print("🔍 Finding JSON files in results/raw/...")
    json_files = get_json_files("results/raw")
    print(f"✓ Found {len(json_files)} JSON files")
    print()
    
    if len(json_files) == 0:
        print("⚠ No JSON files found to archive. Exiting.")
        return
    
    # Archive files
    print("📦 Archiving files...")
    archived_files = archive_files(json_files, archive_dir)
    print()
    
    # Create manifest
    print("📝 Creating manifest...")
    manifest_path = create_manifest(archived_files, archive_dir)
    print()
    
    # Generate comparison baseline
    print("📊 Generating comparison baseline...")
    baseline_path = generate_comparison_baseline(json_files, archive_dir)
    print()
    
    # Delete original files
    print("🗑️  Deleting original files...")
    delete_original_files(json_files)
    print()
    
    print("="*70)
    print("✅ Archival complete!")
    print("="*70)
    print(f"\nArchived {len(archived_files)} files to: {archive_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Baseline: {baseline_path}")
    print("\nYou can now re-run experiments with corrected configurations.")


if __name__ == "__main__":
    main()
