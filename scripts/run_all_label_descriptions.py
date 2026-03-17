#!/usr/bin/env python3
"""Run all 189 label description experiments locally with GPU support.

Usage:
    conda activate zeroshot
    python scripts/run_all_label_descriptions.py

Features:
    - Auto-detects GPU and uses CUDA if available
    - Resumes from where it left off (checks results folder)
    - Logs progress to console and file
    - Continues on errors (logs failed experiments)
    - Estimates remaining time
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Configuration
CONFIG_DIR = Path("src/label_descriptions/experiments")
RESULTS_DIR = Path("results/full_label_descriptions")
LOG_FILE = Path("results/experiment_run.log")
FAILED_LOG = Path("results/failed_experiments.json")

# Ensure results dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(message):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

def get_gpu_info():
    """Get GPU info if available."""
    try:
        result = subprocess.run(
            ["python", "-c", "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"GPU check failed: {e}"

def run_experiment(config_path):
    """Run single experiment and return success/failure."""
    exp_name = config_path.stem
    metrics_file = RESULTS_DIR / f"{exp_name}_metrics.json"
    
    # Skip if already exists
    if metrics_file.exists():
        return True, "already_exists"
    
    cmd = ["python", "main.py", "--config", str(config_path)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        if result.returncode == 0 and metrics_file.exists():
            return True, "success"
        else:
            error_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "timeout (>1 hour)"
    except Exception as e:
        return False, str(e)

def main():
    log("=" * 80)
    log("LABEL DESCRIPTION EXPERIMENTS - FULL RUN")
    log("=" * 80)
    log(f"GPU Info: {get_gpu_info()}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config dir: {CONFIG_DIR.absolute()}")
    log(f"Results dir: {RESULTS_DIR.absolute()}")
    log("")
    
    # Find all config files
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    total = len(config_files)
    log(f"Total experiments to run: {total}")
    
    # Check how many already completed
    already_done = sum(1 for c in config_files if (RESULTS_DIR / f"{c.stem}_metrics.json").exists())
    log(f"Already completed: {already_done}")
    log(f"Remaining: {total - already_done}")
    log("")
    
    # Track results
    completed = 0
    failed = []
    start_time = time.time()
    
    # Run all experiments
    for idx, config_path in enumerate(tqdm(config_files, desc="Experiments"), 1):
        exp_name = config_path.stem
        
        # Check if already done
        if (RESULTS_DIR / f"{exp_name}_metrics.json").exists():
            continue
        
        log(f"[{idx}/{total}] Running: {exp_name}")
        
        success, message = run_experiment(config_path)
        
        if success:
            completed += 1
            log(f"  ✅ {message}")
        else:
            failed.append({"experiment": exp_name, "error": message})
            log(f"  ❌ FAILED: {message}")
            # Save failed list immediately
            with open(FAILED_LOG, "w", encoding="utf-8") as f:
                json.dump(failed, f, indent=2)
        
        # Progress update every 10 experiments
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total - idx)
            log(f"Progress: {idx}/{total} | Elapsed: {elapsed/3600:.1f}h | Est. remaining: {remaining/3600:.1f}h")
            log("")
    
    # Final summary
    elapsed = time.time() - start_time
    log("")
    log("=" * 80)
    log("RUN COMPLETED")
    log("=" * 80)
    log(f"Total experiments: {total}")
    log(f"Already existed: {already_done}")
    log(f"Newly completed: {completed}")
    log(f"Failed: {len(failed)}")
    log(f"Total time: {elapsed/3600:.1f} hours")
    log(f"Average per experiment: {elapsed/total/60:.1f} minutes")
    log(f"Results in: {RESULTS_DIR.absolute()}")
    
    if failed:
        log(f"Failed experiments logged to: {FAILED_LOG.absolute()}")
        log(f"To retry failed experiments, delete their results and re-run this script")
    
    return len(failed)

if __name__ == "__main__":
    sys.exit(main())
