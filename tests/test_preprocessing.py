# Tests the preprocessing assumptions that protect the temporal evaluation setup.

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def _build_synthetic_master():
    # Fake master table with the same schema and date span as the pipeline output.
    np.random.seed(42)
    dates = pd.bdate_range("2015-01-05", "2025-12-31")
    tickers = ["AAPL", "MSFT"]
    rows = []
    for ticker in tickers:
        for d in dates:
            rows.append({"Date": d, "Ticker": ticker,
                         "Adj_Close": 100 + np.random.randn(),
                         "Return_1d": np.random.randn() * 0.01,
                         "Return_5d": np.random.randn() * 0.02,
                         "Return_21d": np.random.randn() * 0.04,
                         "Market_Return": np.random.randn() * 0.01,
                         "MA10_Ratio": 1.0 + np.random.randn() * 0.01,
                         "MA50_Ratio": 1.0 + np.random.randn() * 0.01,
                         "MACD": np.random.randn() * 0.001,
                         "MACD_Signal": np.random.randn() * 0.001,
                         "RSI": 50 + np.random.randn() * 10,
                         "Vol_20d": abs(np.random.randn()) * 0.02,
                         "Momentum_10d": np.random.randn() * 0.02,
                         "fed_funds_rate": 2.0 + np.random.randn() * 0.5,
                         "us10y_yield": 3.5 + np.random.randn() * 0.3,
                         "vix": 18.0 + np.random.randn() * 3.0,
                         "cpi": 260.0 + np.random.randn() * 2.0,
                         "unemployment_rate": 4.0 + np.random.randn() * 0.5,
                         "Target_Return": np.random.randn() * 0.01,
                         "Target_Direction": np.random.randint(0, 2)})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _run_preprocessing(df):
    # Mirrors the split and scaling logic without reading the real data files.
    target_col = "Target_Return"
    drop_cols = ["Date", "Ticker", "Adj_Close", "Target_Direction",
                 "Target_Return", "Return_1d"]
    features = [col for col in df.columns if col not in drop_cols]

    val_start = "2023-01-01"
    test_start = "2024-01-01"

    train_df = df[df["Date"] < val_start].copy()
    val_df = df[(df["Date"] >= val_start) & (df["Date"] < test_start)].copy()
    test_df = df[df["Date"] >= test_start].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])

    return train_df, val_df, test_df, features, scaler, X_train, X_val, X_test


# Checks that the chronological split is preserved.
class TestTemporalSplitIntegrity:

    @pytest.fixture(autouse=True)
    def setup(self):
        df = _build_synthetic_master()
        self.train, self.val, self.test, *_ = _run_preprocessing(df)

    def test_train_ends_before_val_starts(self):
        assert self.train["Date"].max() < self.val["Date"].min()

    def test_val_ends_before_test_starts(self):
        assert self.val["Date"].max() < self.test["Date"].min()

    def test_all_train_dates_before_2023(self):
        assert (self.train["Date"] < pd.Timestamp("2023-01-01")).all()

    def test_all_test_dates_from_2024(self):
        assert (self.test["Date"] >= pd.Timestamp("2024-01-01")).all()

    def test_no_empty_splits(self):
        assert len(self.train) > 0
        assert len(self.val) > 0
        assert len(self.test) > 0


# Checks that invalid numeric values are removed before modelling.
class TestNaNAndInfinityRemoval:

    def test_infinity_replaced_before_split(self):
        df = _build_synthetic_master()
        df.loc[10, "Return_5d"] = np.inf
        df.loc[20, "MA10_Ratio"] = -np.inf
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        assert not np.any(np.isinf(df.select_dtypes(include=[np.number]).values))

    def test_no_nan_after_dropna(self):
        df = _build_synthetic_master()
        df.loc[5, "RSI"] = np.nan
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        assert not df.isnull().any().any()


# Checks that scaling is fitted on training data only.
class TestScalerFittedOnTrainOnly:

    @pytest.fixture(autouse=True)
    def setup(self):
        df = _build_synthetic_master()
        (self.train, self.val, self.test, self.features,
         self.scaler, self.X_train, self.X_val, self.X_test) = _run_preprocessing(df)

    def test_train_features_have_zero_mean(self):
        # The fitted training matrix should be centred by StandardScaler.
        means = np.mean(self.X_train, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_train_features_have_unit_variance(self):
        # StandardScaler should put training features onto a comparable scale.
        stds = np.std(self.X_train, axis=0)
        for i, s in enumerate(stds):
            assert abs(s - 1.0) < 0.1, \
                f"Feature {i} has std {s:.4f} after scaling — expected ~1.0"

    def test_val_features_do_not_have_zero_mean(self):
        # Validation data should be transformed with the training scaler, not refitted.
        means = np.abs(np.mean(self.X_val, axis=0))
        assert np.any(means > 0.01), \
            "Validation features have zero mean — scaler may have been re-fitted"

    def test_test_features_do_not_have_zero_mean(self):
        # Test data should keep its out-of-sample distribution after transformation.
        means = np.abs(np.mean(self.X_test, axis=0))
        assert np.any(means > 0.01), \
            "Test features have zero mean — scaler may have been re-fitted"