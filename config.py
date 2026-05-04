from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent

# Random seeds for reproducibility
RANDOM_SEED = 42

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = DATA_DIR / "modeling"
RESULTS_DIR = DATA_DIR / "results"

# Model directories
MODELS_DIR = ROOT / "models"
BASELINE_DIR = MODELS_DIR / "baselines"
LSTM_DIR = MODELS_DIR / "lstm"

# Raw data paths
PRICES_PATH = RAW_DIR / "prices.csv"
MACRO_PATH = RAW_DIR / "macro_fred.csv"

# Processed data paths
MASTER_DATASET_PATH = PROCESSED_DIR / "master_dataset.csv"

# Modeling paths
X_TRAIN_PATH = MODELING_DIR / "X_train.npy"
X_VAL_PATH = MODELING_DIR / "X_val.npy"
X_TEST_PATH = MODELING_DIR / "X_test.npy"
Y_TRAIN_PATH = MODELING_DIR / "y_train_returns.npy"
Y_VAL_PATH = MODELING_DIR / "y_val_returns.npy"
Y_TEST_PATH = MODELING_DIR / "y_test_returns.npy"

# Results paths
ENSEMBLE_TUNING_LOG = RESULTS_DIR / "ensemble_tuning_log.csv"
HDE_RESULTS_PATH = RESULTS_DIR / "hde_final_results.csv"
LSTM_PREDICTIONS_PATH = RESULTS_DIR / "lstm_predictions.csv"
BEST_ENSEMBLE_CONFIG = RESULTS_DIR / "best_ensemble_config.json"

# Tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
BENCHMARK = "SPY"

# Date boundaries
TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"