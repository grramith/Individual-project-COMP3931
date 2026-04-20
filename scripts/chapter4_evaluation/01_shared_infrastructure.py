import pandas as pd

def run_backtest_per_ticker(preds_df, threshold=0.0, vix_low=18.0, vix_high=22.0,
                             use_fractional=True, allow_short=False, dd_limit=0.15,
                             use_threshold=True, use_vix_filter=True, use_taper=True,
                             tx_cost=TX_COST_DEFAULT, initial_capital=INITIAL_CAPITAL):
    per_ticker = []
    eff_threshold_base = threshold if use_threshold else 0.0

    for ticker in preds_df["Ticker"].unique():
        t = preds_df[preds_df["Ticker"] == ticker].copy().sort_values("Date").reset_index(drop=True)
        n = len(t)
        if n < 2:
            continue

        pred = t["Prediction"].values
        actual = t["Actual"].values
        vix = t["VIX_Value"].values if "VIX_Value" in t.columns else np.zeros(n)

        position = np.zeros(n)
        strat_rets = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = initial_capital
        peak = initial_capital

        for i in range(1, n):
            p = pred[i - 1]
            v = vix[i - 1]

            if use_vix_filter and use_threshold:
                if v > vix_high:
                    eff = eff_threshold_base * 3.0
                elif v > vix_low:
                    eff = eff_threshold_base * 1.5
                else:
                    eff = eff_threshold_base
            else:
                eff = eff_threshold_base

            denom = eff * 5 + 1e-9
            if use_fractional and use_threshold:
                if p > eff:
                    position[i] = min(p / denom, 1.0)
                elif allow_short and p < -eff:
                    position[i] = max(p / denom, -1.0)
                else:
                    position[i] = 0.0
            else:
                if p > eff:
                    position[i] = 1.0
                elif allow_short and p < -eff:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            if use_taper:
                current_dd = (equity[i - 1] - peak) / peak if peak > 0 else 0.0
                if current_dd < -dd_limit:
                    severity = min((abs(current_dd) - dd_limit) / dd_limit, 1.0)
                    position[i] *= max(1.0 - severity, 0.0)

            pos_change = abs(position[i] - position[i - 1])
            ret = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i] = ret
            equity[i] = equity[i - 1] * (1 + ret)
            peak = max(peak, equity[i])

        t["Position"] = position
        t["Strategy_Ret"] = strat_rets
        t["Equity"] = equity
        per_ticker.append(t)

    return per_ticker