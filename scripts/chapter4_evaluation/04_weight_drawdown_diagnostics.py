# Phase 2 - weight diagnosis (Figure 4.1)
# Phase 4 - drawdown decomposition (Table 4.3)

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# Walk up to the project root so this runs regardless of cwd
def _find_project_root():
    sentinel_dirs = {'data', 'scripts', 'models'}
    candidate = os.path.abspath(os.getcwd())
    while True:
        children = {p for p in os.listdir(candidate)
                    if os.path.isdir(os.path.join(candidate, p))}
        if sentinel_dirs <= children:
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            raise RuntimeError("Could not locate project root")
        candidate = parent

PROJECT_ROOT = _find_project_root()
os.chdir(PROJECT_ROOT)

EVAL_DIR = "data/results/evaluation"
TRADING_DAYS = 252
TX_COST_DEFAULT = 0.0005
INITIAL_CAPITAL = 1000.0


# Inferential helpers from script 02 - inlined so this script runs on its own
def select_block_length(x, max_lag=40):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    var0 = np.dot(x, x) / n
    if var0 == 0:
        return 5
    bound = 1.96 / np.sqrt(n)
    for lag in range(1, min(max_lag, n // 4)):
        r = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var0)
        if abs(r) < bound:
            return max(3, min(20, lag))
    return 10


def block_bootstrap(x, statistic, n_boot=10000, block_len=None, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    if block_len is None:
        key = x if x.ndim == 1 else x[:, 0]
        block_len = select_block_length(key)
    p = 1.0 / block_len
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            L = int(rng.geometric(p))
            L = min(L, n - i)
            idx[i:i + L] = (start + np.arange(L)) % n
            i += L
        sample = x[idx]
        boot_stats[b] = statistic(sample)
    point = statistic(x)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])
    return point, (float(lo), float(hi)), boot_stats


def sharpe_difference_test(r1, r2, periods=TRADING_DAYS):
    r1, r2 = np.asarray(r1), np.asarray(r2)
    n = min(len(r1), len(r2))
    r1, r2 = r1[-n:], r2[-n:]
    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    if s1 == 0 or s2 == 0:
        return {"sr1": 0.0, "sr2": 0.0, "diff": 0.0, "z": 0.0, "p_value": 1.0}
    sr1_d = mu1 / s1
    sr2_d = mu2 / s2
    corr = np.corrcoef(r1, r2)[0, 1]
    var = (1 / n) * (
        2 - 2 * corr +
        0.5 * (sr1_d ** 2 + sr2_d ** 2 - 2 * sr1_d * sr2_d * corr ** 2)
    )
    var = max(var, 1e-12)
    z = (sr1_d - sr2_d) / np.sqrt(var)
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    sr1_ann = sr1_d * np.sqrt(periods)
    sr2_ann = sr2_d * np.sqrt(periods)
    return {
        "sr1": float(sr1_ann),
        "sr2": float(sr2_ann),
        "diff": float(sr1_ann - sr2_ann),
        "z": float(z),
        "p_value": float(p),
    }


def sharpe_annualised(rets, periods=TRADING_DAYS):
    rets = np.asarray(rets)
    if len(rets) == 0 or np.std(rets, ddof=1) == 0:
        return 0.0
    return (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(periods)


# Backtest engine from script 01 - inlined so this script runs on its own
def run_backtest(
    preds_df,
    threshold=0.0,
    vix_low=18.0,
    vix_high=22.0,
    use_fractional=True,
    allow_short=False,
    dd_limit=0.15,
    use_threshold=True,
    use_vix_filter=True,
    use_taper=True,
    tx_cost=TX_COST_DEFAULT,
    initial_capital=INITIAL_CAPITAL,
):
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

    if not per_ticker:
        return None

    combined = pd.concat(per_ticker, ignore_index=True)
    port = combined.groupby("Date").agg(
        Actual=("Actual", "mean"),
        Strategy_Ret=("Strategy_Ret", "mean"),
        Position=("Position", "mean"),
    ).reset_index().sort_values("Date").reset_index(drop=True)
    port["Equity"] = initial_capital * (1 + port["Strategy_Ret"]).cumprod()
    port["Market_Cum"] = initial_capital * (1 + port["Actual"]).cumprod()

    rets = port["Strategy_Ret"].values
    mkt = port["Actual"].values

    def _max_dd(equity):
        equity = np.asarray(equity)
        peak = np.maximum.accumulate(equity)
        return float(((equity - peak) / peak).min())

    def _sortino(rets, periods=TRADING_DAYS):
        downside = rets[rets < 0]
        if len(downside) == 0 or downside.std(ddof=1) == 0:
            return 0.0
        return (rets.mean() / downside.std(ddof=1)) * np.sqrt(periods)

    def _calmar(rets, equity, periods=TRADING_DAYS):
        ann_ret = rets.mean() * periods
        mdd = abs(_max_dd(equity))
        return 0.0 if mdd == 0 else ann_ret / mdd

    stats = {
        "total_return_pct": (port["Equity"].iloc[-1] / initial_capital - 1) * 100,
        "sharpe": sharpe_annualised(rets),
        "sortino": _sortino(rets),
        "max_drawdown": _max_dd(port["Equity"].values),
        "calmar": _calmar(rets, port["Equity"].values),
        "avg_exposure": float(np.mean(port["Position"])),
        "n_days": len(rets),
        "market_total_return_pct": (port["Market_Cum"].iloc[-1] / initial_capital - 1) * 100,
        "market_sharpe": sharpe_annualised(mkt),
        "market_max_drawdown": _max_dd(port["Market_Cum"].values),
    }

    return {
        "per_ticker": per_ticker,
        "combined": combined,
        "portfolio": port,
        "stats": stats,
        "daily_returns": rets,
        "market_returns": mkt,
    }


# Load shared state from earlier scripts
with open(f"{EVAL_DIR}/_state_phase0.pkl", "rb") as f:
    _phase0 = pickle.load(f)
PREDS = _phase0["PREDS"]
HDE_CONFIG = _phase0["HDE_CONFIG"]

with open(f"{EVAL_DIR}/_state_phase3.pkl", "rb") as f:
    _phase3 = pickle.load(f)
STRATEGIES = _phase3["STRATEGIES"]


# Strategy helpers reused from script 03
def preds_for_model(col_name):
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value", col_name]].copy()
    df.rename(columns={col_name: "Prediction"}, inplace=True)
    return df.dropna(subset=["Prediction"])


def run_strategy(label, preds_df, **override):
    kwargs = dict(
        threshold=HDE_CONFIG["threshold"],
        vix_low=HDE_CONFIG["vix_low"],
        vix_high=HDE_CONFIG["vix_high"],
        use_fractional=HDE_CONFIG.get("fractional", True),
        allow_short=HDE_CONFIG.get("allow_short", False),
        dd_limit=HDE_CONFIG["dd_limit"],
    )
    kwargs.update(override)
    result = run_backtest(preds_df, **kwargs)
    result["label"] = label
    return result


# Figure 4.1 - weight trajectories and deviation stats
def weight_diagnosis():
    print("\n" + "=" * 78)
    print("Section 4.5 — DYNAMIC WEIGHTING MECHANISM DIAGNOSIS")
    print("=" * 78)

    hde = pd.read_csv("data/results/hde_final_results.csv", parse_dates=["Date"])
    w_cols = ["Weight_RF", "Weight_GB", "Weight_LSTM"]

    # One row per date - average across tickers
    daily = hde.groupby("Date")[w_cols].mean().sort_index()

    print("\nTime-series statistics of daily portfolio-level weights:")
    stats = pd.DataFrame({
        "mean": daily.mean(),
        "std":  daily.std(),
        "min":  daily.min(),
        "max":  daily.max(),
    })
    stats["max_abs_dev_from_uniform"] = (daily - 1/3).abs().max()
    stats["frac_within_±0.05"] = ((daily - 1/3).abs() < 0.05).mean()
    print(stats.round(4).to_string())

    # How often all three weights stay near 1/3
    all_within = ((daily - 1/3).abs() < 0.05).all(axis=1).mean()
    print(f"\nFraction of days ALL THREE weights within ±0.05 of 1/3:  {all_within:.2%}")
    print(f"Max abs deviation of any weight on any day from 1/3:     "
          f"{(daily - 1/3).abs().values.max():.4f}")

    # Figure 4.1 - the three weight series with the 1/3 reference and ±0.05 band
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily["Weight_RF"], label="Random Forest", color="#3b82f6", lw=1.5)
    ax.plot(daily.index, daily["Weight_GB"], label="Gradient Boosting", color="#ef4444", lw=1.5)
    ax.plot(daily.index, daily["Weight_LSTM"], label="LSTM", color="#10b981", lw=1.5)
    ax.axhline(1/3, color="black", ls="--", alpha=0.5, label="Uniform (1/3)")
    ax.fill_between(daily.index, 1/3 - 0.05, 1/3 + 0.05,
                    color="gray", alpha=0.15, label="±0.05 band")
    ax.set_title("Ensemble weight trajectories across the test period",
                 fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0.2, 0.45)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/figure_4_1_weight_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {EVAL_DIR}/figure_4_1_weight_trajectories.png")

    stats.to_csv(f"{EVAL_DIR}/weight_diagnostics.csv")

    # HDE vs equal-weight - if p > 0.05 the dynamic weights aren't doing anything
    print("\nCritical test — does the dynamic weighting add anything?")
    hde_rets = STRATEGIES["e_HDE"]["daily_returns"]
    eq_rets = STRATEGIES["d_EqualWeight"]["daily_returns"]
    hde_port = STRATEGIES["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "hde"})
    eq_port = STRATEGIES["d_EqualWeight"]["portfolio"][["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "eq"})
    merged = hde_port.merge(eq_port, on="Date", how="inner")

    test = sharpe_difference_test(merged["hde"].values, merged["eq"].values)
    print(f"  HDE Sharpe:        {test['sr1']:+.3f}")
    print(f"  Equal-wt Sharpe:   {test['sr2']:+.3f}")
    print(f"  Δ Sharpe:          {test['diff']:+.3f}")
    print(f"  JKM z-statistic:   {test['z']:+.3f}")
    print(f"  p-value:           {test['p_value']:.4f}")
    if test["p_value"] > 0.05:
        print("  → HDE and equal-weight ensemble are STATISTICALLY INDISTINGUISHABLE")
        print("     The dynamic weighting mechanism is empirically inert.")
    elif test["diff"] > 0:
        print("  → HDE significantly outperforms equal-weight ensemble")
    else:
        print("  → Equal-weight ensemble significantly outperforms HDE")
        print("     The dynamic weighting is actively harmful.")

    # Bootstrap the return difference directly as a sanity check on JKM
    diff_rets = merged["hde"].values - merged["eq"].values
    bl = select_block_length(diff_rets)
    mean_diff, (lo, hi), _ = block_bootstrap(diff_rets, np.mean,
                                              n_boot=5000, block_len=bl)
    print(f"\n  Bootstrap mean daily return difference (HDE − EqualWt):")
    print(f"  point={mean_diff*1e4:+.2f} bps  CI=[{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps")

    return daily, stats, test


WEIGHTS, WEIGHT_STATS, HDE_VS_EQUAL_TEST = weight_diagnosis()


# Phase 4 - 2x2x2 ablation of the three defensive components
def drawdown_decomposition():
    print("\n" + "=" * 78)
    print("TABLE 4.3 — Drawdown Decomposition (full 2×2×2 factorial)")
    print("=" * 78)
    print("All eight combinations of the three defensive components on HDE predictions.")
    print("Base configuration uses the tuned HDE parameters from Section 4.3.\n")

    hde_preds = preds_for_model("Pred_HDE")
    rows = []
    for use_thresh in [False, True]:
        for use_vix in [False, True]:
            for use_tap in [False, True]:
                res = run_strategy(
                    f"T={int(use_thresh)}/V={int(use_vix)}/D={int(use_tap)}",
                    hde_preds,
                    use_threshold=use_thresh,
                    use_vix_filter=use_vix,
                    use_taper=use_tap,
                )
                s = res["stats"]
                rows.append({
                    "Threshold": use_thresh,
                    "VIX_filter": use_vix,
                    "Taper": use_tap,
                    "Sharpe": round(s["sharpe"], 3),
                    "Max_DD_%": round(s["max_drawdown"] * 100, 2),
                    "Total_Return_%": round(s["total_return_pct"], 1),
                    "Exposure_%": round(s["avg_exposure"] * 100, 1),
                })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Average effect of each component when enabled vs disabled
    print("\nMarginal contribution to drawdown improvement")
    print("(averaged across all configurations of the other two components):")

    def marginal_effect(component):
        # positive = turning it on reduces drawdown
        on = table[table[component] == True]["Max_DD_%"].mean()
        off = table[table[component] == False]["Max_DD_%"].mean()
        return off - on

    for comp in ["Threshold", "VIX_filter", "Taper"]:
        effect = marginal_effect(comp)
        print(f"  {comp:<12}  ΔMax_DD when enabled: {effect:+.2f} pp  "
              f"({'reduces' if effect > 0 else 'worsens'} drawdown)")

    # All-on vs all-off
    full = table[(table["Threshold"]) & (table["VIX_filter"]) & (table["Taper"])]
    none = table[(~table["Threshold"]) & (~table["VIX_filter"]) & (~table["Taper"])]
    if len(full) and len(none):
        total_improvement = none.iloc[0]["Max_DD_%"] - full.iloc[0]["Max_DD_%"]
        print(f"\n  Total drawdown reduction (all-on vs all-off): {total_improvement:+.2f} pp")

    table.to_csv(f"{EVAL_DIR}/table_4_3_drawdown_decomposition.csv", index=False)
    return table


TABLE_4_3 = drawdown_decomposition()


# Save outputs so script 05 can pick them up
with open(f"{EVAL_DIR}/_state_phase4.pkl", "wb") as f:
    pickle.dump({
        "WEIGHTS": WEIGHTS,
        "WEIGHT_STATS": WEIGHT_STATS,
        "HDE_VS_EQUAL_TEST": HDE_VS_EQUAL_TEST,
        "TABLE_4_3": TABLE_4_3,
    }, f)
print(f"saved phase-4 state to {EVAL_DIR}/_state_phase4.pkl")