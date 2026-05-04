# Tests the ensemble assumptions that affect the Chapter 4 HDE results.

import numpy as np
import pandas as pd
import pytest


# Local copy of the single-step weighting logic used by the HDE.
def compute_ensemble_weights_single_step(
    hist_act, hist_rf, hist_gb, hist_lstm,
    p_rf, p_gb, p_lstm,
    window, decay, alpha, prev_sw,
    eps=1e-6
):
    decay_w = np.array([decay ** (window - 1 - i) for i in range(window)])
    decay_w /= decay_w.sum()

    # Estimate recent forecast bias before combining the model outputs.
    bias_rf = float(np.dot(decay_w, hist_rf - hist_act))
    bias_gb = float(np.dot(decay_w, hist_gb - hist_act))
    pred_rf_c = p_rf - bias_rf
    pred_gb_c = p_gb - bias_gb

    # Score each available model using recent magnitude and direction errors.
    ew_mae_rf = float(np.dot(decay_w, np.abs(hist_rf - hist_act)))
    ew_mae_gb = float(np.dot(decay_w, np.abs(hist_gb - hist_act)))
    dir_rf = float(np.dot(decay_w, ((hist_rf > 0) == (hist_act > 0)).astype(float)))
    dir_gb = float(np.dot(decay_w, ((hist_gb > 0) == (hist_act > 0)).astype(float)))
    score_rf = 0.7 / (ew_mae_rf + eps) + 0.3 * dir_rf
    score_gb = 0.7 / (ew_mae_gb + eps) + 0.3 * dir_gb

    # During LSTM warm-up, fall back to the two tabular models only.
    if np.any(np.isnan(hist_lstm)):
        total = score_rf + score_gb
        raw_rf, raw_gb, raw_lstm = score_rf / total, score_gb / total, 0.0
        pred_lstm_c = 0.0
    else:
        bias_lstm = float(np.dot(decay_w, hist_lstm - hist_act))
        pred_lstm_c = p_lstm - bias_lstm
        ew_mae_lstm = float(np.dot(decay_w, np.abs(hist_lstm - hist_act)))
        dir_lstm = float(np.dot(decay_w, ((hist_lstm > 0) == (hist_act > 0)).astype(float)))
        score_lstm = 0.7 / (ew_mae_lstm + eps) + 0.3 * dir_lstm
        total = score_rf + score_gb + score_lstm
        raw_rf = score_rf / total
        raw_gb = score_gb / total
        raw_lstm = score_lstm / total

    # Smooth weight updates so one noisy window cannot dominate immediately.
    sw_rf = (1 - alpha) * prev_sw["rf"] + alpha * raw_rf
    sw_gb = (1 - alpha) * prev_sw["gb"] + alpha * raw_gb
    sw_lstm = (1 - alpha) * prev_sw["lstm"] + alpha * raw_lstm
    sw_sum = sw_rf + sw_gb + sw_lstm

    weights = {
        "rf": sw_rf / sw_sum,
        "gb": sw_gb / sw_sum,
        "lstm": sw_lstm / sw_sum,
    }
    smoothed = {"rf": sw_rf, "gb": sw_gb, "lstm": sw_lstm}
    preds_corrected = {"rf": pred_rf_c, "gb": pred_gb_c, "lstm": pred_lstm_c}

    return weights, smoothed, preds_corrected


# Tests


# Checks that the adaptive weights remain a valid convex combination.
class TestWeightsSumToOne:

    @pytest.mark.parametrize("decay", [0.90, 0.95, 1.0])
    @pytest.mark.parametrize("alpha", [0.05, 0.15, 0.50])
    def test_weights_sum_to_one(self, decay, alpha):
        np.random.seed(42)
        window = 10
        hist_act = np.random.randn(window) * 0.01
        hist_rf = hist_act + np.random.randn(window) * 0.005
        hist_gb = hist_act + np.random.randn(window) * 0.005
        hist_lstm = hist_act + np.random.randn(window) * 0.005

        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        weights, _, _ = compute_ensemble_weights_single_step(
            hist_act, hist_rf, hist_gb, hist_lstm,
            p_rf=0.001, p_gb=-0.001, p_lstm=0.0005,
            window=window, decay=decay, alpha=alpha, prev_sw=prev_sw,
        )
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-8, \
            f"Weights sum to {total}, not 1.0 (decay={decay}, alpha={alpha})"

    def test_weights_sum_to_one_after_many_steps(self):
        # Repeated updates should not introduce numerical drift in the weights.
        np.random.seed(42)
        window = 10
        n = 100 + window
        actuals = np.random.randn(n) * 0.01
        preds_rf = actuals + np.random.randn(n) * 0.005
        preds_gb = actuals + np.random.randn(n) * 0.005
        preds_lstm = actuals + np.random.randn(n) * 0.005

        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        for t in range(window, n):
            weights, smoothed, _ = compute_ensemble_weights_single_step(
                actuals[t - window:t],
                preds_rf[t - window:t],
                preds_gb[t - window:t],
                preds_lstm[t - window:t],
                p_rf=preds_rf[t], p_gb=preds_gb[t], p_lstm=preds_lstm[t],
                window=window, decay=0.95, alpha=0.15,
                prev_sw=prev_sw,
            )
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-8, \
                f"Step {t}: weights sum to {total}"
            prev_sw = smoothed


# Checks the warm-up fallback used before LSTM predictions are available.
class TestLSTMFallback:

    def test_lstm_weight_is_zero_when_unavailable(self):
        np.random.seed(42)
        window = 10
        hist_act = np.random.randn(window) * 0.01
        hist_rf = hist_act + np.random.randn(window) * 0.005
        hist_gb = hist_act + np.random.randn(window) * 0.005
        hist_lstm = np.full(window, np.nan)

        # alpha=1 isolates the fallback behaviour without smoothing carry-over.
        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        weights, _, preds = compute_ensemble_weights_single_step(
            hist_act, hist_rf, hist_gb, hist_lstm,
            p_rf=0.001, p_gb=-0.001, p_lstm=np.nan,
            window=window, decay=0.95, alpha=1.0,
            prev_sw=prev_sw,
        )
        assert weights["lstm"] == 0.0, \
            f"LSTM weight should be 0.0 when unavailable, got {weights['lstm']}"

    def test_rf_gb_renormalise_when_lstm_unavailable(self):
        np.random.seed(42)
        window = 10
        hist_act = np.random.randn(window) * 0.01
        hist_rf = hist_act + np.random.randn(window) * 0.005
        hist_gb = hist_act + np.random.randn(window) * 0.005
        hist_lstm = np.full(window, np.nan)

        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        weights, _, _ = compute_ensemble_weights_single_step(
            hist_act, hist_rf, hist_gb, hist_lstm,
            p_rf=0.001, p_gb=-0.001, p_lstm=np.nan,
            window=window, decay=0.95, alpha=1.0,
            prev_sw=prev_sw,
        )
        rf_gb_sum = weights["rf"] + weights["gb"]
        assert abs(rf_gb_sum - 1.0) < 1e-8, \
            f"RF + GB should sum to 1.0 when LSTM unavailable, got {rf_gb_sum}"


# Checks that the residual bias adjustment does not move unbiased forecasts.
class TestBiasCorrection:

    def test_zero_bias_on_perfect_predictions(self):
        window = 10
        hist_act = np.array([0.01, -0.005, 0.003, -0.002, 0.007,
                             0.001, -0.004, 0.006, -0.001, 0.002])
        hist_rf = hist_act.copy()
        hist_gb = hist_act.copy()
        hist_lstm = hist_act.copy()

        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        _, _, preds_corrected = compute_ensemble_weights_single_step(
            hist_act, hist_rf, hist_gb, hist_lstm,
            p_rf=0.005, p_gb=0.005, p_lstm=0.005,
            window=window, decay=0.95, alpha=0.15, prev_sw=prev_sw,
        )
        # Perfect historical predictions should leave the next predictions unchanged.
        assert abs(preds_corrected["rf"] - 0.005) < 1e-10
        assert abs(preds_corrected["gb"] - 0.005) < 1e-10
        assert abs(preds_corrected["lstm"] - 0.005) < 1e-10


# Checks that EMA smoothing stops weights from jumping too sharply.
class TestEMASmoothingBounds:

    def test_weight_change_bounded_by_alpha(self):
        # Make RF clearly best so the test stresses the smoothing step.
        np.random.seed(42)
        window = 10
        hist_act = np.random.randn(window) * 0.01

        hist_rf = hist_act + np.random.randn(window) * 0.0001
        hist_gb = hist_act + np.random.randn(window) * 0.05
        hist_lstm = hist_act + np.random.randn(window) * 0.05

        alpha = 0.15
        prev_sw = {"rf": 1 / 3, "gb": 1 / 3, "lstm": 1 / 3}
        weights, _, _ = compute_ensemble_weights_single_step(
            hist_act, hist_rf, hist_gb, hist_lstm,
            p_rf=0.001, p_gb=0.001, p_lstm=0.001,
            window=window, decay=0.95, alpha=alpha, prev_sw=prev_sw,
        )
        # A single noisy window should not move any model close to an extreme weight.
        for model in ["rf", "gb", "lstm"]:
            change = abs(weights[model] - 1 / 3)
            assert change < 0.5, \
                f"Weight for {model} changed by {change:.4f} in one step — " \
                f"EMA smoothing may not be working"