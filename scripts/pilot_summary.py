"""Pilot summary: compare L1/L2/L3 results for AG News x BGE from results/label_semantics/."""

import json
import os
from pathlib import Path

RESULTS_DIR = Path("results/label_semantics")

# Expected files: experiment_name -> file stem
EXPERIMENTS = {
    "pilot_ag_news_bge_name_only": "L1 (name_only)",
    "pilot_ag_news_bge_description": "L2 (description)",
    "ag_news_bge_multi_description": "L3 (multi_description)",
}


def load_metrics(exp_name: str) -> dict | None:
    path = RESULTS_DIR / f"{exp_name}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print(f"\n{'='*65}")
    print("  AG News x BGE-M3 — Pilot Label Mode Comparison")
    print(f"{'='*65}")
    print(f"  Results dir: {RESULTS_DIR}\n")

    header = f"  {'Experiment':<35} {'Label Mode':<22} {'Macro-F1':>9} {'Accuracy':>9}"
    print(header)
    print(f"  {'-'*63}")

    for exp_name, label_mode in EXPERIMENTS.items():
        metrics = load_metrics(exp_name)
        if metrics is None:
            print(f"  {'[MISSING] ' + exp_name:<35} {label_mode:<22} {'N/A':>9} {'N/A':>9}")
            continue
        macro_f1 = metrics.get("macro_f1", float("nan"))
        accuracy = metrics.get("accuracy", float("nan"))
        print(f"  {exp_name:<35} {label_mode:<22} {macro_f1:>9.4f} {accuracy:>9.4f}")

    print(f"\n  Note: Run missing experiments before comparing.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
