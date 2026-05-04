# Tests model output shapes and the LSTM training checks used in the pipeline.

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Same LSTM structure as the training script.
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


N_TRAIN = 500
N_FEATURES = 16
SEQ_LEN = 10
BATCH_SIZE = 32


@pytest.fixture
def flat_data():
    # Flat arrays match the input expected by the sklearn regressors.
    np.random.seed(42)
    X = np.random.randn(N_TRAIN, N_FEATURES)
    y = np.random.randn(N_TRAIN)
    return X, y


@pytest.fixture
def sequence_data():
    # Sequence tensors match the LSTM input format used after windowing.
    torch.manual_seed(42)
    X = torch.randn(N_TRAIN, SEQ_LEN, N_FEATURES)
    y = torch.randn(N_TRAIN)
    return X, y


# Checks that each sklearn model returns one prediction per row.
class TestSklearnOutputShapes:

    def test_linear_regression_shape(self, flat_data):
        X, y = flat_data
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N_TRAIN,)

    def test_ridge_regression_shape(self, flat_data):
        X, y = flat_data
        model = Ridge(alpha=1.0).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N_TRAIN,)

    def test_random_forest_shape(self, flat_data):
        X, y = flat_data
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N_TRAIN,)

    def test_gradient_boosting_shape(self, flat_data):
        X, y = flat_data
        model = GradientBoostingRegressor(n_estimators=10, random_state=42).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (N_TRAIN,)


# Checks the tensor conventions expected by the LSTM.
class TestLSTMDimensions:

    def test_input_tensor_shape(self, sequence_data):
        X, y = sequence_data
        assert X.shape == (N_TRAIN, SEQ_LEN, N_FEATURES), \
            f"Expected (batch, seq_len, features), got {X.shape}"

    def test_output_shape_matches_batch(self, sequence_data):
        X, y = sequence_data
        model = LSTMRegressor(input_size=N_FEATURES, hidden_size=64)
        model.eval()
        with torch.no_grad():
            preds = model(X[:BATCH_SIZE])
        assert preds.shape == (BATCH_SIZE,), \
            f"LSTM output shape {preds.shape} does not match batch size {BATCH_SIZE}"

    def test_output_is_scalar_per_sample(self, sequence_data):
        # The return target is one next-day value per sequence.
        X, _ = sequence_data
        model = LSTMRegressor(input_size=N_FEATURES, hidden_size=128)
        model.eval()
        with torch.no_grad():
            preds = model(X[:1])
        assert preds.ndim == 1 and preds.shape[0] == 1


# Checks that early stopping keeps the best validation state.
class TestEarlyStopping:

    def test_best_weights_restored_after_patience(self, sequence_data):
        # The restored model should match the best recorded validation loss.
        X, y = sequence_data
        torch.manual_seed(42)
        model = LSTMRegressor(input_size=N_FEATURES, hidden_size=32, dropout=0.0)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 3
        patience_counter = 0

        for epoch in range(50):
            model.train()
            optimiser.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X), y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(best_state)
                    break

        model.eval()
        with torch.no_grad():
            restored_loss = criterion(model(X), y).item()

        assert abs(restored_loss - best_val_loss) < 1e-5, \
            f"Restored loss {restored_loss:.6f} != best loss {best_val_loss:.6f} — " \
            "early stopping did not restore best weights"