"""Master script to regenerate all figures, tables, and analyses.

Runs all analysis scripts in the correct dependency order with progress
logging and timing information. Run from the project root:

    python scripts/regenerate_all.py

Optional flags:
    --skip-slow     Skip computationally expensive steps (task characteristics,
                    confusion matrices) if results already exist.
    --results-dir   Path to results directory (default: results/raw)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Step definitions
# Each step is (label, script_path, description)
# Steps are executed in order; later steps may depend on earlier ones.
# ---------------------------------------------------------------------------
STEPS = [
    (
        "collect_results",
        "scripts/collect_all_results.py",
        "Collect all raw JSON results into summary CSV",
    ),
    (
        "statistical_analysis",
        "scripts/statistical_analysis.py",
        "Friedman test + Nemenyi post-hoc + power analysis",
    ),
    (
        "publication_heatmap",
        "scripts/generate_publication_heatmap.py",
        "Model × dataset Macro-F1 heatmap (PDF/EPS)",
    ),
    (
        "critical_difference",
        "scripts/generate_critical_difference_diagram.py",
        "Critical difference diagram (PDF/EPS)",
    ),
    (
        "label_formulation",
        "scripts/generate_label_formulation_figure.py",
        "Label formulation comparison figure (name_only vs description)",
    ),
    (
        "task_type_analysis",
        "scripts/generate_task_type_analysis.py",
        "Task type grouped bar chart (PDF/EPS)",
    ),
    (
        "stability_compute",
        "scripts/analyze_model_stability.py",
        "Compute model stability metrics (coefficient of variation)",
    ),
    (
        "stability_visualize",
        "scripts/visualize_model_stability.py",
        "Stability vs performance scatter plot (PDF/EPS)",
    ),
    (
        "task_characteristics_compute",
        "scripts/analyze_task_characteristics.py",
        "Compute task characteristics (num_classes, text_length, label_similarity)",
    ),
    (
        "task_characteristics_visualize",
        "scripts/visualize_task_characteristics.py",
        "Task characteristics scatter plots (PDF/EPS)",
    ),
    (
        "error_patterns",
        "scripts/analyze_error_patterns.py",
        "Analyze error patterns and most-confused class pairs",
    ),
    (
        "confusion_matrices",
        "scripts/generate_confusion_matrices.py",
        "Confusion matrix heatmaps for representative datasets",
    ),
]

# Steps that are slow and can be skipped if outputs already exist
SLOW_STEPS = {"task_characteristics_compute", "confusion_matrices", "error_patterns"}


def _fmt_duration(seconds: float) -> str:
    """Format duration as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _log(msg: str) -> None:
    """Print a timestamped log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run_step(label: str, script: str, description: str, step_num: int, total: int) -> tuple[bool, float]:
    """Run a single analysis step.

    Returns:
        (success, elapsed_seconds)
    """
    _log(f"Step {step_num}/{total}: {description}")

    script_path = Path(script)
    if not script_path.exists():
        _log(f"  WARNING: Script not found — {script} (skipping)")
        return False, 0.0

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,  # let output stream to terminal
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        _log(f"  done ({_fmt_duration(elapsed)})")
        return True, elapsed
    else:
        _log(f"  FAILED (exit code {result.returncode}, {_fmt_duration(elapsed)})")
        return False, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all figures, tables, and analyses."
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow steps (task characteristics, confusion matrices) if outputs exist",
    )
    parser.add_argument(
        "--results-dir",
        default="results/raw",
        help="Path to raw results directory (default: results/raw)",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    for d in ["results/tables", "results/plots", "results/statistical_analysis",
              "results/stability_analysis", "results/task_characteristics"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    steps_to_run = []
    for label, script, description in STEPS:
        if args.skip_slow and label in SLOW_STEPS:
            _log(f"Skipping slow step: {description}")
            continue
        steps_to_run.append((label, script, description))

    total = len(steps_to_run)
    _log(f"Starting full analysis regeneration ({total} steps)")
    print("-" * 60, flush=True)

    wall_start = time.perf_counter()
    results: list[tuple[str, bool, float]] = []

    for i, (label, script, description) in enumerate(steps_to_run, start=1):
        success, elapsed = _run_step(label, script, description, i, total)
        results.append((label, success, elapsed))
        print(flush=True)

    wall_elapsed = time.perf_counter() - wall_start

    # Summary
    print("=" * 60, flush=True)
    _log(f"Regeneration complete. Total time: {_fmt_duration(wall_elapsed)}")
    print()

    passed = [(l, e) for l, ok, e in results if ok]
    failed = [(l, e) for l, ok, e in results if not ok]

    print(f"  Passed: {len(passed)}/{total}")
    if failed:
        print(f"  Failed: {len(failed)}/{total}")
        for label, elapsed in failed:
            print(f"    - {label}")

    print()
    print("Output locations:")
    print("  Figures:              results/plots/")
    print("  Tables:               results/tables/")
    print("  Statistical analysis: results/statistical_analysis/")
    print("  Stability analysis:   results/stability_analysis/")
    print("  Task characteristics: results/task_characteristics/")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
