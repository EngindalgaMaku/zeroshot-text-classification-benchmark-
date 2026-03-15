"""
Verify seed implementation in main.py and related files.

This script checks:
1. set_seed(42) is called before config loading
2. CUDA deterministic mode is enabled when GPU available
3. Seed value logging to experiment output files

Requirements: 1.1, 1.2, 1.4, 1.5
"""

import ast
import sys
from pathlib import Path


def check_main_py():
    """Verify seed implementation in main.py."""
    print("="*70)
    print("Verifying Seed Implementation in main.py")
    print("="*70)
    print()
    
    main_path = Path("main.py")
    if not main_path.exists():
        print("❌ ERROR: main.py not found")
        return False
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "set_seed function exists": False,
        "set_seed(42) called in main": False,
        "set_seed called before load_config": False,
        "CUDA deterministic mode enabled": False,
        "Seed value logged to console": False
    }
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Check for set_seed function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed":
            checks["set_seed function exists"] = True
            
            # Check for CUDA deterministic mode
            func_content = ast.get_source_segment(content, node)
            if "torch.backends.cudnn.deterministic = True" in func_content:
                checks["CUDA deterministic mode enabled"] = True
    
    # Check main function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_content = ast.get_source_segment(content, node)
            
            # Check if set_seed(42) is called
            if "set_seed(42)" in main_content:
                checks["set_seed(42) called in main"] = True
            
            # Check if set_seed is called before load_config
            set_seed_pos = main_content.find("set_seed(42)")
            load_config_pos = main_content.find("load_config")
            if set_seed_pos != -1 and load_config_pos != -1 and set_seed_pos < load_config_pos:
                checks["set_seed called before load_config"] = True
            
            # Check for seed logging
            if "Random seed: 42" in main_content or "seed: 42" in main_content:
                checks["Seed value logged to console"] = True
    
    # Print results
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    print()
    return all_passed


def check_runner_py():
    """Verify seed is logged to experiment output files."""
    print("="*70)
    print("Verifying Seed Logging in runner.py")
    print("="*70)
    print()
    
    runner_path = Path("src/runner.py")
    if not runner_path.exists():
        print("❌ ERROR: src/runner.py not found")
        return False
    
    with open(runner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if random_seed is added to metrics
    if 'metrics["random_seed"]' in content or "metrics['random_seed']" in content:
        print("✅ Seed value is logged to experiment output files")
        print("   Found: metrics['random_seed'] = 42")
        print()
        return True
    else:
        print("❌ Seed value is NOT logged to experiment output files")
        print("   Expected: metrics['random_seed'] = 42")
        print()
        return False


def main():
    """Main verification process."""
    print("\n")
    print("="*70)
    print("SEED IMPLEMENTATION VERIFICATION")
    print("="*70)
    print()
    
    main_check = check_main_py()
    runner_check = check_runner_py()
    
    print("="*70)
    if main_check and runner_check:
        print("✅ ALL CHECKS PASSED")
        print("="*70)
        print()
        print("Summary:")
        print("  ✓ set_seed(42) is called before config loading")
        print("  ✓ CUDA deterministic mode is enabled when GPU available")
        print("  ✓ Seed value is logged to console output")
        print("  ✓ Seed value is logged to experiment output files")
        print()
        print("Requirements satisfied: 1.1, 1.2, 1.4, 1.5")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*70)
        print()
        print("Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
