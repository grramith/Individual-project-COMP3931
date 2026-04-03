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
]

# Simple visual separator for readability
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
