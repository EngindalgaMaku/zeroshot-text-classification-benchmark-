"""Compare banking77 experiment results."""
import json
from pathlib import Path

results_dir = Path("results/raw")

experiments = {
    "Manual (description)": "exp_banking77_mpnet_metrics.json",
    "Generated L2": "exp_banking77_mpnet_l2_metrics.json",
    "Generated L3": "exp_banking77_mpnet_l3_metrics.json",
}

print("=== Banking77 Results Comparison ===\n")
print(f"{'Method':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
print("-" * 65)

for name, filename in experiments.items():
    filepath = results_dir / filename
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        acc = data.get('accuracy', 0)
        macro_f1 = data.get('macro_f1', 0)
        weighted_f1 = data.get('weighted_f1', 0)
        print(f"{name:<25} {acc:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f}")
    else:
        print(f"{name:<25} {'NOT RUN':<12} {'NOT RUN':<12} {'NOT RUN':<12}")

print("\n" + "=" * 65)
