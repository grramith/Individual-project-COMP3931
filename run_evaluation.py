import subprocess
import sys

# Chapter 4 evaluation pipeline - run after main.py has produced the HDE outputs
scripts = [
    "scripts/chapter4_evaluation/01_shared_infrastructure.py",
    "scripts/chapter4_evaluation/02_inferential_toolbox.py",
    "scripts/chapter4_evaluation/03_predictive_performance.py",
    "scripts/chapter4_evaluation/04_weight_drawdown_diagnostics.py",
    "scripts/chapter4_evaluation/05_regime_robustness_summary.py",
    "scripts/chapter4_evaluation/06_enhanced_backtest.py",
]

SEPARATOR = "=" * 80

for script in scripts:
    print(f"\n{SEPARATOR}")
    print(f"Running: {script}")
    print(f"{SEPARATOR}")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\n{SEPARATOR}")
        print(f"Stopped because {script} failed.")
        print(f"{SEPARATOR}")
        sys.exit(result.returncode)

    print(f"\nFinished: {script}")
    print(f"{SEPARATOR}")


print("Evaluation pipeline complete.")