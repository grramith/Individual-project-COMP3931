# Tests the evaluation metrics against simple hand-checkable cases.

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Local copies of the metric functions used in the evaluation scripts.
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)


def directional_accuracy(y_true, y_pred):
    return np.mean((y_pred > 0) == (y_true > 0))


def sharpe_ratio(returns, periods=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


# Checks absolute-error behaviour.
class TestMAE:

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        # Three 0.5 errors should average to 0.5.
        assert abs(mae(y_true, y_pred) - 0.5) < 1e-10

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 1.0, 4.0])
        # MAE should only depend on error size, not direction.
        assert abs(mae(y_true, y_pred) - 1.0) < 1e-10


# Checks squared-error behaviour.
class TestRMSE:

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        # Only the final point is wrong, so the RMSE is easy to verify.
        expected = np.sqrt(4.0 / 3.0)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_rmse_geq_mae(self):
        # Squaring should make RMSE at least as large as MAE.
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


# Checks the R-squared cases used to interpret weak return forecasts.
class TestRSquared:

    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        assert abs(r_squared(y_true, y_true) - 1.0) < 1e-10

    def test_mean_prediction_gives_zero(self):
        # Predicting the sample mean should be the zero-skill baseline.
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert abs(r_squared(y_true, y_pred) - 0.0) < 1e-10

    def test_negative_r2_worse_than_mean(self):
        # This is the case that matters for several return-prediction baselines.
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        assert r_squared(y_true, y_pred) < 0.0


# Checks the sign-based accuracy metric used for trading interpretation.
class TestDirectionalAccuracy:

    def test_perfect_direction(self):
        y_true = np.array([0.01, -0.02, 0.03, -0.01])
        y_pred = np.array([0.005, -0.01, 0.02, -0.005])
        assert directional_accuracy(y_true, y_pred) == 1.0

    def test_all_wrong_direction(self):
        y_true = np.array([0.01, -0.02, 0.03])
        y_pred = np.array([-0.01, 0.02, -0.03])
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_known_mixed_accuracy(self):
        y_true = np.array([0.01, -0.02, 0.03, -0.01])
        y_pred = np.array([0.005, 0.01, 0.02, -0.005])
        # Three of the four signs match.
        assert abs(directional_accuracy(y_true, y_pred) - 0.75) < 1e-10

    def test_zero_predictions_counted(self):
        # This documents the exact sign convention used by the pipeline.
        y_true = np.array([-0.01, -0.02])
        y_pred = np.array([0.0, 0.0])
        assert directional_accuracy(y_true, y_pred) == 1.0


# Checks the annualised Sharpe helper used in the results tables.
class TestSharpeRatio:

    def test_zero_std_returns_zero(self):
        # Constant returns are treated as undefined Sharpe, returned as 0.0.
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        assert sharpe_ratio(returns) == 0.0

    def test_known_value(self):
        # Same formula as the evaluation script: mean divided by std, annualised.
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        mean_r = returns.mean()
        std_r = returns.std()
        expected = (mean_r / std_r) * np.sqrt(252)
        assert abs(sharpe_ratio(returns) - expected) < 1e-10

    def test_negative_sharpe_for_losing_strategy(self):
        returns = np.array([-0.01, -0.02, -0.005, -0.015, 0.001])
        assert sharpe_ratio(returns) < 0.0

    def test_all_flat_returns_zero(self):
        # All-flat returns should follow the same zero-volatility convention.
        returns = np.zeros(100)
        assert sharpe_ratio(returns) == 0.0