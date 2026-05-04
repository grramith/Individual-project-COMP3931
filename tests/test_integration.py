import pandas as pd
import numpy as np
import os
import pytest


def test_pipeline_outputs_exist():
    # These files are the hand-off points between the main pipeline stages.
    expected_outputs = [
        "data/raw/prices.csv",
        "data/raw/macro_fred.csv",
        "data/processed/master_dataset.csv",
        "data/modeling/X_train.npy",
        "data/modeling/X_val.npy",
        "data/modeling/X_test.npy",
        "data/results/baseline_regression_results.csv",
        "data/results/hyperparameter_tuning_log.csv",
        "data/results/lstm_predictions.csv",
        "data/results/lstm_tuning_log.csv",
        "data/results/hde_final_results.csv",
        "data/results/ensemble_tuning_log.csv",
    ]
    for path in expected_outputs:
        assert os.path.exists(path), f"Missing output: {path}"
        assert os.path.getsize(path) > 0, f"Empty output: {path}"


def test_pipeline_data_integrity():
    # The final HDE output should be readable and contain usable prediction rows.
    results = pd.read_csv("data/results/hde_final_results.csv")
    assert "Date" in results.columns
    assert "Ticker" in results.columns
    assert len(results) > 0
    assert not results.isnull().all().any()


def test_ensemble_output_integrity():
    # The tuning log should cover the full grid used for the selected HDE setup.
    tuning_log = pd.read_csv("data/results/ensemble_tuning_log.csv")
    assert len(tuning_log) == 432, f"Expected 432 configs, got {len(tuning_log)}"


def test_temporal_integrity_across_stages():
    # Train, validation, and test metadata must stay in chronological order.
    train_meta = pd.read_csv("data/modeling/train_metadata.csv")
    val_meta = pd.read_csv("data/modeling/val_metadata.csv")
    test_meta = pd.read_csv("data/modeling/test_metadata.csv")

    assert train_meta["Date"].max() < val_meta["Date"].min()
    assert val_meta["Date"].max() < test_meta["Date"].min()