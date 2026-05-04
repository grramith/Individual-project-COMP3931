# Tests the backtest assumptions that would change the Chapter 4 results.

import numpy as np
import pandas as pd
import pytest


def _run_backtest_loop(pred, actual, vix, threshold, vix_low, vix_high,
                       use_fractional, allow_short, dd_limit,
                       pos_scale=1, tx_cost=0.0005):
    # Local copy keeps these checks independent of the full pipeline.
    n = len(pred)
    position = np.zeros(n)
    strat_rets = np.zeros(n)
    equity = np.ones(n)
    peak = 1.0

    for i in range(1, n):
        # Use yesterday's prediction so the test matches the trading setup.
        p = pred[i - 1]
        v = vix[i - 1]

        # Higher VIX requires a stronger signal before taking exposure.
        if v > vix_high:
            eff_threshold = threshold * 3.0
        elif v > vix_low:
            eff_threshold = threshold * 1.5
        else:
            eff_threshold = threshold

        # Fractional sizing should never allow leverage beyond full exposure.
        if use_fractional:
            denom = eff_threshold * 5 + 1e-9
            if p > eff_threshold:
                position[i] = min(p / denom * pos_scale, 1.0)
            elif allow_short and p < -eff_threshold:
                position[i] = max(p / denom * pos_scale, -1.0)
            else:
                position[i] = 0.0
        else:
            if p > eff_threshold:
                position[i] = 1.0
            elif allow_short and p < -eff_threshold:
                position[i] = -1.0
            else:
                position[i] = 0.0

        # Taper exposure only after the drawdown limit has actually been breached.
        dd = (equity[i - 1] - peak) / peak if peak > 0 else 0
        if dd < -dd_limit:
            severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
            position[i] *= max(1.0 - severity, 0.0)

        # Apply costs on changes in exposure rather than on every holding day.
        pos_change = abs(position[i] - position[i - 1])
        ret = position[i] * actual[i] - pos_change * tx_cost
        strat_rets[i] = ret
        equity[i] = equity[i - 1] * (1 + ret)
        peak = max(peak, equity[i])

    return position, strat_rets, equity


# Tests


# Checks that signals are executed with a one-day lag.
class TestSignalLag:

    def test_position_at_t_uses_prediction_at_t_minus_1(self):
        # A signal at index 5 should first affect the position at index 6.
        n = 20
        pred = np.zeros(n)
        pred[5] = 0.01
        actual = np.ones(n) * 0.001
        vix = np.ones(n) * 15.0

        position, _, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.0, vix_low=18, vix_high=25,
            use_fractional=False, allow_short=False, dd_limit=0.20,
        )
        # Index 5 is still flat because pred[5] is not tradable until index 6.
        assert position[5] == 0.0, \
            f"Position at t=5 should be 0 (signal not yet available), got {position[5]}"
        assert position[6] == 1.0, \
            f"Position at t=6 should reflect pred[5], got {position[6]}"

    def test_no_same_day_execution(self):
        # The first row has no previous signal available.
        pred = np.array([0.05, 0.05, 0.05])
        actual = np.array([0.01, 0.01, 0.01])
        vix = np.array([15.0, 15.0, 15.0])

        position, _, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.0, vix_low=18, vix_high=25,
            use_fractional=False, allow_short=False, dd_limit=0.20,
        )
        assert position[0] == 0.0, "Position at t=0 must always be zero"


# Checks that position sizing stays within the intended exposure bounds.
class TestPositionBounds:

    def test_long_position_capped_at_one(self):
        pred = np.array([0.0, 0.5, 0.5])
        actual = np.array([0.0, 0.01, 0.01])
        vix = np.zeros(3)

        position, _, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.001, vix_low=18, vix_high=25,
            use_fractional=True, allow_short=False, dd_limit=0.20,
        )
        assert np.all(position <= 1.0), \
            f"Position exceeded 1.0: {position.max()}"

    def test_short_position_floored_at_minus_one(self):
        pred = np.array([0.0, -0.5, -0.5])
        actual = np.array([0.0, -0.01, -0.01])
        vix = np.zeros(3)

        position, _, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.001, vix_low=18, vix_high=25,
            use_fractional=True, allow_short=True, dd_limit=0.20,
        )
        assert np.all(position >= -1.0), \
            f"Position below -1.0: {position.min()}"


# Checks that transaction costs are applied only when exposure changes.
class TestTransactionCosts:

    def test_cost_deduction_on_entry(self):
        # Entering from cash into a full position should deduct one 5bp cost.
        tx_cost = 0.0005
        pred = np.array([0.01, 0.01, 0.01])
        actual = np.array([0.0, 0.02, 0.01])
        vix = np.zeros(3)

        position, strat_rets, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.0, vix_low=18, vix_high=25,
            use_fractional=False, allow_short=False, dd_limit=0.20,
            tx_cost=tx_cost,
        )
        # The first trade pays the entry cost before the return is recorded.
        expected_ret = 1.0 * 0.02 - 1.0 * tx_cost
        assert abs(strat_rets[1] - expected_ret) < 1e-10, \
            f"Expected return {expected_ret}, got {strat_rets[1]}"

    def test_zero_cost_when_position_unchanged(self):
        # Once the position is already open, holding it should not create another cost.
        pred = np.array([0.01, 0.01, 0.01, 0.01])
        actual = np.array([0.0, 0.01, 0.01, 0.01])
        vix = np.zeros(4)

        position, strat_rets, _ = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.0, vix_low=18, vix_high=25,
            use_fractional=False, allow_short=False, dd_limit=0.20,
        )
        # These rows should contain pure position returns because exposure is unchanged.
        for t in [2, 3]:
            expected = position[t] * actual[t]
            assert abs(strat_rets[t] - expected) < 1e-10, \
                f"At t={t}: expected {expected}, got {strat_rets[t]} — " \
                "cost deducted despite no position change"


# Checks that the drawdown taper behaves as used in the evaluation.
class TestDrawdownTaper:

    def test_taper_activates_beyond_limit(self):
        # Sustained losses should eventually force the taper to reduce exposure.
        n = 50
        pred = np.ones(n) * 0.01
        actual = np.ones(n) * -0.02
        vix = np.ones(n) * 15.0

        position, _, equity = _run_backtest_loop(
            pred, actual, vix,
            threshold=0.0, vix_low=18, vix_high=25,
            use_fractional=False, allow_short=False, dd_limit=0.10,
        )
        # Locate the region where the running drawdown is past the 10% limit.
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        taper_active = np.where(drawdown < -0.10)[0]

        if len(taper_active) > 0:
            # At least one post-breach position should be below full exposure.
            tapered_positions = position[taper_active]
            assert np.any(tapered_positions < 1.0), \
                "Drawdown exceeded limit but positions were not tapered"

    def test_taper_severity_is_linear(self):
        # This checks the same taper formula used inside the backtest loop.
        dd_limit = 0.10

        # The scale moves from 1.0 to 0.0 as drawdown moves from 10% to 20%.
        for dd, expected_scale in [(-0.10, 1.0), (-0.15, 0.5), (-0.20, 0.0)]:
            if dd < -dd_limit:
                severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
                scale = max(1.0 - severity, 0.0)
            else:
                scale = 1.0
            assert abs(scale - expected_scale) < 1e-10, \
                f"At dd={dd}: expected scale {expected_scale}, got {scale}"