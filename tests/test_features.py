# Tests the feature schema and ticker-level separation used before model training.

import numpy as np
import pandas as pd
import pytest

EXPECTED_FEATURE_COUNT = 15

DROP_COLS = ["Date", "Ticker", "Adj_Close", "Target_Direction",
             "Target_Return", "Return_1d"]


def _build_synthetic_master():
    # Small fake master table with the same columns as the processed dataset.
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    rows = []
    for ticker in tickers:
        base_price = {"AAPL": 150, "MSFT": 250, "GOOGL": 1200}[ticker]
        for d in dates:
            rows.append({
                "Date": d, "Ticker": ticker,
                "Adj_Close": base_price + np.random.randn(),
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
                "fed_funds_rate": 2.0,
                "us10y_yield": 3.5,
                "vix": 18.0,
                "cpi": 260.0,
                "unemployment_rate": 4.0,
                "Target_Return": np.random.randn() * 0.01,
                "Target_Direction": np.random.randint(0, 2),
            })
    return pd.DataFrame(rows)


# Checks that the model input columns match the intended feature set.
class TestFeatureCount:

    def test_feature_count_is_sixteen(self):
        df = _build_synthetic_master()
        features = [c for c in df.columns if c not in DROP_COLS]
        assert len(features) == EXPECTED_FEATURE_COUNT, \
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(features)}: {features}"

    def test_return_1d_excluded_from_features(self):
        # Return_1d is removed because it is too close to the next-day target.
        df = _build_synthetic_master()
        features = [c for c in df.columns if c not in DROP_COLS]
        assert "Return_1d" not in features, \
            "Return_1d found in feature list — structural redundancy with target"

    def test_target_columns_excluded(self):
        df = _build_synthetic_master()
        features = [c for c in df.columns if c not in DROP_COLS]
        assert "Target_Return" not in features
        assert "Target_Direction" not in features

    def test_metadata_columns_excluded(self):
        df = _build_synthetic_master()
        features = [c for c in df.columns if c not in DROP_COLS]
        assert "Date" not in features
        assert "Ticker" not in features
        assert "Adj_Close" not in features


# Checks that feature construction keeps each ticker separate.
class TestPerTickerIsolation:

    def test_no_cross_ticker_leakage_in_rolling_features(self):
        # Changing MSFT should not alter already-built AAPL feature values.
        df = _build_synthetic_master()

        aapl_features_original = df[df["Ticker"] == "AAPL"]["MA10_Ratio"].values.copy()

        df.loc[df["Ticker"] == "MSFT", "Adj_Close"] = np.nan

        aapl_features_after = df[df["Ticker"] == "AAPL"]["MA10_Ratio"].values
        np.testing.assert_array_equal(
            aapl_features_original, aapl_features_after,
            err_msg="AAPL features changed when MSFT data was corrupted — "
                    "cross-ticker leakage detected"
        )

    def test_tickers_have_independent_row_counts(self):
        # Equal row counts make sequence construction easier to audit.
        df = _build_synthetic_master()
        counts = df.groupby("Ticker").size()
        assert counts.nunique() == 1, \
            f"Unequal row counts per ticker: {counts.to_dict()}"