import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM for daily return prediction.
    Dropout is applied between LSTM layers and before the output
    to reduce overfitting on noisy financial data.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)

def create_sequences_per_ticker(X, y, metadata, seq_len):
    """
    Create sequences within each ticker separately so windows
    do not mix rows from different stocks.
    """
    sequences, targets, meta_rows = [], [], []
    tickers = metadata["Ticker"].unique()

    for ticker in tickers:
        mask = metadata["Ticker"].values == ticker
        X_tick = X[mask]
        y_tick = y[mask]
        meta_tick = metadata[mask].reset_index(drop=True)

        for i in range(seq_len, len(X_tick)):
            sequences.append(X_tick[i - seq_len:i])
            targets.append(y_tick[i])
            meta_rows.append({
                "Date": meta_tick.iloc[i]["Date"],
                "Ticker": ticker
            })

    return (
        np.array(sequences),
        np.array(targets),
        pd.DataFrame(meta_rows)
    )

def train_lstm():
    # Load modelling arrays
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")
    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    # Load metadata so sequences can be built per ticker
    meta_train = pd.read_csv("data/modeling/train_metadata.csv")
    meta_val = pd.read_csv("data/modeling/val_metadata.csv")
    meta_test = pd.read_csv("data/modeling/test_metadata.csv")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    hyperparam_configs = [
        {"seq_len": 10, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 20, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 10, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
        {"seq_len": 20, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
    ]

    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    PATIENCE = 10

    print("Tuning LSTM hyperparameters on validation set...")

if __name__ == "__main__":
    train_lstm()