import subprocess
import sys

# Define pipeline order so everything runs sequentially
scripts = [
    "scripts/01_data_collection.py",
    "scripts/02_feature_engineering.py",
    "scripts/03_build_master_dataset.py",
    "scripts/04_regression_data_preprocessing.py",
    "scripts/05_train_baseline_regressors.py",
    "scripts/06_train_lstm_regressor.py",
    "scripts/07_build_enhanced_hde.py",

    # Chapter 4 evaluation - run after the HDE is built and predictions are saved
    "scripts/chapter4_evaluation/01_shared_infrastructure.py",
    "scripts/chapter4_evaluation/02_inferential_toolbox.py",
    "scripts/chapter4_evaluation/03_predictive_performance.py",
    "scripts/chapter4_evaluation/04_weight_drawdown_diagnostics.py",
    "scripts/chapter4_evaluation/05_regime_robustness_summary.py",
    "scripts/chapter4_evaluation/06_enhanced_backtest.py",
]

# Simple visual separator for readability, purposefully made choice
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


print("Pipeline complete.")
