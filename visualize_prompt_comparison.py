"""Visualize the comparison between old and new prompt results."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Results data
metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Precision', 'Macro Recall']

# L2 results
l2_old = [60.5, 59.3, 59.8, 66.4, 59.3]
l2_new = [57.5, 55.9, 56.7, 63.4, 57.1]

# L3 results
l3_old = [57.2, 54.5, 55.3, 61.0, 56.8]
l3_new = [56.3, 53.5, 54.4, 60.2, 56.1]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# L2 comparison
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, l2_old, width, label='Old (15-20 words)', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, l2_new, width, label='New (10-15 words)', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('L2 Descriptions Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([50, 70])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# L3 comparison
bars3 = ax2.bar(x - width/2, l3_old, width, label='Old (15-20 words)', color='#2ecc71', alpha=0.8)
bars4 = ax2.bar(x + width/2, l3_new, width, label='New (10-15 words)', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Score (%)', fontsize=12)
ax2.set_title('L3 Descriptions Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([50, 70])

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/prompt_optimization_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: results/prompt_optimization_comparison.png")

# Create word count comparison
fig2, ax = plt.subplots(figsize=(10, 6))

labels_sample = ['Label 0\n(activate)', 'Label 2\n(apple_pay)', 'Label 7\n(beneficiary)', 
                 'Label 10\n(acceptance)', 'Label 14\n(not_working)', 'Average\n(all 77)']
old_counts = [15, 15, 18, 18, 18, 16.7]
new_counts = [8, 13, 11, 12, 12, 10.3]

x = np.arange(len(labels_sample))
width = 0.35

bars1 = ax.bar(x - width/2, old_counts, width, label='Old (15-20 words)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, new_counts, width, label='New (10-15 words)', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Word Count', fontsize=12)
ax.set_title('L2 Description Length Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_sample)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# Add reduction percentages
for i, (old, new) in enumerate(zip(old_counts, new_counts)):
    reduction = ((old - new) / old) * 100
    ax.text(i, max(old, new) + 1, f'-{reduction:.0f}%', 
            ha='center', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('results/word_count_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Word count visualization saved to: results/word_count_comparison.png")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"L2 Accuracy: {l2_old[0]:.1f}% → {l2_new[0]:.1f}% (Δ {l2_new[0]-l2_old[0]:.1f}%)")
print(f"L3 Accuracy: {l3_old[0]:.1f}% → {l3_new[0]:.1f}% (Δ {l3_new[0]-l3_old[0]:.1f}%)")
print(f"Avg Words:   {old_counts[-1]:.1f} → {new_counts[-1]:.1f} (Δ {new_counts[-1]-old_counts[-1]:.1f})")
print("="*60)
print("Conclusion: Shorter descriptions led to WORSE performance")
print("="*60)
