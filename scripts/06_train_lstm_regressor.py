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
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")
    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

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

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None
    tuning_log = []

    print("Tuning LSTM hyperparameters on validation set...")
    print("=" * 70)

    for cfg in hyperparam_configs:
        seq_len = cfg["seq_len"]
        print(f"\nConfig: {cfg}")

        X_tr_seq, y_tr_seq, _ = create_sequences_per_ticker(X_train, y_train, meta_train, seq_len)
        X_val_seq, y_val_seq, _ = create_sequences_per_ticker(X_val, y_val, meta_val, seq_len)

        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            print(f"  Skipping: insufficient data for seq_len={seq_len}")
            continue

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_seq),
            torch.FloatTensor(y_tr_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_size = X_train.shape[1]
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=2,
            dropout=cfg["dropout"]
        ).to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.MSELoss()

        best_epoch_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)

            val_loss /= len(val_dataset)

            if val_loss < best_epoch_val_loss:
                best_epoch_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        tuning_log.append({
            "config": str(cfg),
            "best_val_loss": best_epoch_val_loss,
            "stopped_epoch": epoch + 1
        })

        print(f"  >> Best Val Loss: {best_epoch_val_loss:.6f}")

        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_config = cfg
            best_model_state = best_state

    if best_config is None:
        print("ERROR: No valid configuration found!")
        return None, None

    print(f"\n{'=' * 70}")
    print(f"Best LSTM Config: {best_config}")
    print(f"Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_lstm()