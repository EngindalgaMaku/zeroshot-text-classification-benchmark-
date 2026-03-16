"""Generate confusion matrices for error analysis.

This script generates confusion matrices for representative datasets
(AG News, Banking77, GoEmotions) across all models.

**Validates: Requirements 10.1**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from labels import LABEL_SETS

# Publication-quality styling
sns.set_style("white")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Representative datasets for confusion matrix analysis
REPRESENTATIVE_DATASETS = ["ag_news", "banking77", "go_emotions", "imdb", "sst2"]

# Model names mapping
MODEL_NAMES = {
    "bge": "BGE-M3",
    "e5": "E5-large",
    "instructor": "Instructor",
    "jina_v5": "Jina-v5",
    "mpnet": "MPNet",
    "nomic": "Nomic",
    "qwen3": "Qwen3"
}

# Dataset display names
DATASET_NAMES = {
    "ag_news": "AG News",
    "banking77": "Banking77",
    "go_emotions": "GoEmotions",
    "imdb": "IMDB",
    "sst2": "SST-2",
}


def load_predictions(dataset, model):
    """Load prediction CSV file for a dataset-model combination."""
    pred_file = Path(f"results/raw/{dataset}_{model}_predictions.csv")
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def get_label_names(dataset):
    """Get label names for a dataset."""
    # Map dataset names to label set keys
    label_key_map = {
        "ag_news": "ag_news",
        "banking77": "banking77",
        "go_emotions": "go_emotions",
        "imdb": "imdb",
        "sst2": "sst2",
    }
    
    label_key = label_key_map.get(dataset)
    if label_key and label_key in LABEL_SETS:
        labels = LABEL_SETS[label_key]["name_only"]
        # Extract just the first label text for each class
        return [labels[i][0] for i in sorted(labels.keys())]
    return None


def create_confusion_matrix_plot(y_true, y_pred, labels, dataset_name, model_name, output_path):
    """Create and save a publication-quality confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels) to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Determine figure size based on number of classes
    n_classes = len(labels)
    if n_classes <= 4:
        figsize = (8, 6)
        font_size = 10
    elif n_classes <= 10:
        figsize = (10, 8)
        font_size = 9
    elif n_classes <= 30:
        figsize = (14, 12)
        font_size = 7
    else:
        figsize = (18, 16)
        font_size = 6
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5, linecolor='gray',
                ax=ax, annot_kws={'size': font_size})
    
    ax.set_title(f'{dataset_name} - {model_name}\nConfusion Matrix (Row-Normalized %)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    return cm, cm_normalized


def main():
    """Generate confusion matrices for representative datasets."""
    print("=" * 80)
    print("GENERATING CONFUSION MATRICES FOR ERROR ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("results/plots/confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Representative datasets: {', '.join(REPRESENTATIVE_DATASETS)}")
    print(f"🤖 Models: {', '.join(MODEL_NAMES.values())}")
    
    # Track statistics
    total_generated = 0
    missing_files = []
    
    # Generate confusion matrices for each dataset-model combination
    for dataset in REPRESENTATIVE_DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {DATASET_NAMES[dataset]}")
        print(f"{'='*80}")
        
        # Get label names
        labels = get_label_names(dataset)
        if labels is None:
            print(f"   ⚠️  Warning: No labels found for {dataset}")
            continue
        
        print(f"   Classes: {len(labels)}")
        
        for model_key, model_name in MODEL_NAMES.items():
            # Load predictions
            df = load_predictions(dataset, model_key)
            
            if df is None:
                missing_files.append(f"{dataset}_{model_key}")
                print(f"   ⚠️  {model_name}: Prediction file not found")
                continue
            
            # Generate confusion matrix
            output_file = output_dir / f"{dataset}_{model_key}_confusion_matrix.pdf"
            
            try:
                cm, cm_norm = create_confusion_matrix_plot(
                    df['y_true'].values,
                    df['y_pred'].values,
                    labels,
                    DATASET_NAMES[dataset],
                    model_name,
                    output_file
                )
                
                # Calculate accuracy from confusion matrix
                accuracy = np.trace(cm) / np.sum(cm) * 100
                
                print(f"   ✅ {model_name}: {output_file.name} (Accuracy: {accuracy:.1f}%)")
                total_generated += 1
                
            except Exception as e:
                print(f"   ❌ {model_name}: Error - {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Generated {total_generated} confusion matrices")
    print(f"📁 Saved to: {output_dir}")
    
    if missing_files:
        print(f"\n⚠️  Missing prediction files ({len(missing_files)}):")
        for f in missing_files[:10]:  # Show first 10
            print(f"   - {f}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    
    print("\n✅ Confusion matrix generation complete!")


if __name__ == "__main__":
    main()
