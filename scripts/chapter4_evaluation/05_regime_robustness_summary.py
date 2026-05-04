# Phase 5 - regime regression (Section 4.4)
# Phase 6 - per-ticker alpha + tx-cost sensitivity
# Phase 7 - Figures 4.2 and 4.3
# Final - chapter summary report

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# Walk up to find the project root so this runs from any cwd
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


# Inferential helpers (inlined from script 02)
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


def sharpe_annualised(rets, periods=TRADING_DAYS):
    rets = np.asarray(rets)
    if len(rets) == 0 or np.std(rets, ddof=1) == 0:
        return 0.0
    return (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(periods)


# Backtest engine (inlined from script 01)
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


# Pull state from earlier scripts
with open(f"{EVAL_DIR}/_state_phase0.pkl", "rb") as f:
    _phase0 = pickle.load(f)
PREDS = _phase0["PREDS"]
HDE_CONFIG = _phase0["HDE_CONFIG"]

with open(f"{EVAL_DIR}/_state_phase3.pkl", "rb") as f:
    _phase3 = pickle.load(f)
STRATEGIES = _phase3["STRATEGIES"]
TABLE_4_1 = _phase3["TABLE_4_1"]
DM_MATRIX = _phase3["DM_MATRIX"]
LADDER_PVALS = _phase3["LADDER_PVALS"]

with open(f"{EVAL_DIR}/_state_phase4.pkl", "rb") as f:
    _phase4 = pickle.load(f)
HDE_VS_EQUAL_TEST = _phase4["HDE_VS_EQUAL_TEST"]
TABLE_4_3 = _phase4["TABLE_4_3"]


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


# Generate walk-forward results inline if not already cached
def generate_walk_forward_if_missing():
    # GB stand-in for the full HDE - retraining HDE in every window would take hours
    wf_path = "data/results/rolling_window_evaluation.csv"
    if os.path.exists(wf_path):
        print(f"  Walk-forward CSV already exists at {wf_path}")
        return pd.read_csv(wf_path, parse_dates=["Window_Start"])

    print("  Walk-forward CSV not found — generating now (≈2-3 min)")
    print("  Approach: GradientBoostingRegressor retrained on each expanding window")
    print("            (matches the methodology of the existing pipeline)")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    df = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    target_col = "Target_Return"
    drop_cols = ["Date", "Ticker", "Adj_Close", "Target_Direction",
                 "Target_Return", "Return_1d"]
    features = [c for c in df.columns if c not in drop_cols]

    # Eight semi-annual windows starting Jan 2021
    eval_start_dates = pd.to_datetime([
        "2021-01-01", "2021-07-01",
        "2022-01-01", "2022-07-01",
        "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01",
    ])
    WINDOW_SIZE_DAYS = int(126 * 1.5)

    rows = []
    for start in eval_start_dates:
        end = start + pd.Timedelta(days=WINDOW_SIZE_DAYS)
        train = df[df["Date"] < start]
        test = df[(df["Date"] >= start) & (df["Date"] < end)]
        if len(test) == 0 or len(train) == 0:
            continue

        # Refit per window so the scaler only sees train data
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[features].values)
        Xte = scaler.transform(test[features].values)
        ytr = train[target_col].values
        yte = test[target_col].values

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)

        mae = mean_absolute_error(yte, preds)
        dir_acc = float(np.mean((preds > 0) == (yte > 0)))
        rows.append({
            "Window_Start": start,
            "Train_Size": len(train),
            "Test_Size": len(test),
            "MAE": mae,
            "Dir_Accuracy": dir_acc,
        })
        print(f"    {start.date()}: train={len(train):>6}  test={len(test):>5}  "
              f"MAE={mae:.6f}  DirAcc={dir_acc:.2%}")

    wf = pd.DataFrame(rows)
    wf.to_csv(wf_path, index=False)
    print(f"\n  Saved → {wf_path}")
    print(f"  Mean DirAcc: {wf['Dir_Accuracy'].mean():.2%}  "
          f"(σ = {wf['Dir_Accuracy'].std():.2%})")
    print(f"  Mean MAE:    {wf['MAE'].mean():.6f}  "
          f"(σ = {wf['MAE'].std():.6f})")
    return wf


def regime_regression():
    print("\n" + "=" * 78)
    print("Section 4.4 — WALK-FORWARD + REGIME REGRESSION")
    print("=" * 78)

    wf = generate_walk_forward_if_missing()
    if wf is None or len(wf) == 0:
        print("  [error] walk-forward generation failed")
        return None, None

    master = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    ret_col = "Return_1d" if "Return_1d" in master.columns else "Target_Return"
    vix_col = [c for c in master.columns if "vix" in c.lower()][0]

    # Term spread is optional - skip if not in the dataset
    term_col = None
    for cand in ["Term_Spread", "Yield_Spread", "T10Y3M", "t10y3m_spread"]:
        if cand in master.columns:
            term_col = cand
            break

    regime_rows = []
    WINDOW_DAYS = int(126 * 1.5)  # matches Script 09
    for _, r in wf.iterrows():
        start = pd.Timestamp(r["Window_Start"])
        end = start + pd.Timedelta(days=WINDOW_DAYS)
        win = master[(master["Date"] >= start) & (master["Date"] < end)]
        if len(win) == 0:
            continue

        daily = win.groupby("Date").agg({
            ret_col: "mean",
            vix_col: "mean",
        }).reset_index()

        realised_vol = daily[ret_col].std() * np.sqrt(252)
        mean_vix = daily[vix_col].mean()
        cum_return = (1 + daily[ret_col]).prod() - 1

        # Average pairwise correlation across the seven tickers
        wide = win.pivot_table(index="Date", columns="Ticker", values=ret_col)
        corr_mat = wide.corr()
        avg_corr = (corr_mat.values[np.triu_indices_from(corr_mat.values, k=1)]).mean()

        row = {
            "Window_Start": r["Window_Start"],
            "DirAcc": r["Dir_Accuracy"],
            "MAE": r["MAE"],
            "Mean_VIX": mean_vix,
            "Realised_Vol": realised_vol,
            "Avg_Pair_Corr": avg_corr,
            "Cum_Return": cum_return,
        }
        if term_col:
            row["Term_Spread"] = win[term_col].mean()
        regime_rows.append(row)

    regimes = pd.DataFrame(regime_rows)
    print("\nPer-window regime features:")
    print(regimes.round(4).to_string(index=False))

    # n=8, so treat these as exploratory
    print("\nUnivariate OLS: DirAcc ~ regime_var   (n = {})".format(len(regimes)))
    candidate_vars = ["Mean_VIX", "Realised_Vol", "Avg_Pair_Corr", "Cum_Return"]
    if term_col:
        candidate_vars.append("Term_Spread")

    uni_rows = []
    for v in candidate_vars:
        x = regimes[v].values
        y = regimes["DirAcc"].values
        if len(x) < 3 or np.std(x) == 0:
            continue
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        uni_rows.append({
            "Variable": v,
            "Coefficient": round(slope, 6),
            "Std_Error": round(se, 6),
            "R_squared": round(r ** 2, 4),
            "p_value": round(p, 4),
        })
    uni_table = pd.DataFrame(uni_rows).sort_values("R_squared", ascending=False)
    print(uni_table.to_string(index=False))

    print("\nCaveat: n = {} windows. These p-values are exploratory and should".format(len(regimes)))
    print("not be interpreted as confirmatory evidence. Report as motivation for")
    print("a regime-conditional architecture (Future Work Section 4.7).")

    regimes.to_csv(f"{EVAL_DIR}/regime_features.csv", index=False)
    uni_table.to_csv(f"{EVAL_DIR}/regime_regression.csv", index=False)
    return regimes, uni_table


REGIMES, REGIME_REGRESSION = regime_regression() or (None, None)


# Per-ticker alpha vs buy & hold, then a cross-sectional t-test
def per_ticker_alpha():
    print("\n" + "=" * 78)
    print("Section 4.3.3 — Per-ticker alpha cross-sectional test")
    print("=" * 78)
    hde = STRATEGIES["e_HDE"]["combined"]
    bh = STRATEGIES["a_BuyHold"]["combined"]

    rows = []
    for ticker in hde["Ticker"].unique():
        h = hde[hde["Ticker"] == ticker]
        b = bh[bh["Ticker"] == ticker]
        m = h[["Date", "Strategy_Ret"]].merge(
            b[["Date", "Strategy_Ret"]], on="Date", suffixes=("_hde", "_bh"))
        alpha_series = m["Strategy_Ret_hde"] - m["Strategy_Ret_bh"]
        mean_alpha_ann = alpha_series.mean() * 252
        t_stat, p_val = sp_stats.ttest_1samp(alpha_series, 0.0)
        rows.append({
            "Ticker": ticker,
            "Annualised_Alpha_%": round(mean_alpha_ann * 100, 2),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Is the average alpha distinguishable from zero?
    alphas = table["Annualised_Alpha_%"].values
    t_cs, p_cs = sp_stats.ttest_1samp(alphas, 0.0)
    print(f"\nCross-sectional mean alpha: {alphas.mean():.2f}%  "
          f"t={t_cs:.3f}  p={p_cs:.4f}")
    if p_cs > 0.05:
        print("  → Mean per-ticker alpha is NOT distinguishable from zero.")
        print("     The HDE's cross-sectional contribution is not statistically")
        print("     separable from a zero-alpha strategy after accounting for")
        print("     cross-sectional variance.")

    table.to_csv(f"{EVAL_DIR}/per_ticker_alpha.csv", index=False)
    return table


PER_TICKER_ALPHA = per_ticker_alpha()


# Sweep tx costs to find the break-even point
def tx_cost_sensitivity():
    print("\n" + "=" * 78)
    print("Section 4.3 — Transaction cost sensitivity (HDE)")
    print("=" * 78)
    hde_preds = preds_for_model("Pred_HDE")
    rows = []
    for bps in [0, 5, 10, 15, 20, 30]:
        res = run_strategy(f"{bps} bps", hde_preds, tx_cost=bps / 10000)
        s = res["stats"]
        rows.append({
            "TX_cost_bps": bps,
            "Total_Return_%": round(s["total_return_pct"], 1),
            "Sharpe": round(s["sharpe"], 3),
            "Max_DD_%": round(s["max_drawdown"] * 100, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(f"{EVAL_DIR}/tx_cost_sensitivity.csv", index=False)
    return table


TX_SENSITIVITY = tx_cost_sensitivity()


# Phase 7 - chapter figures
def build_display_items():
    print("\n" + "=" * 78)
    print("Phase 7 — Display items")
    print("=" * 78)

    # Figure 4.2 - 60-day rolling Sharpe across the three headline strategies
    hde_port = STRATEGIES["e_HDE"]["portfolio"]
    eq_port = STRATEGIES["d_EqualWeight"]["portfolio"]
    bh_port = STRATEGIES["a_BuyHold"]["portfolio"]

    def rolling_sharpe(s, window=60):
        return s.rolling(window).mean() / s.rolling(window).std() * np.sqrt(252)

    merged = hde_port[["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "HDE"}).merge(
        eq_port[["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "EqualWt"}),
        on="Date", how="inner").merge(
        bh_port[["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "BuyHold"}),
        on="Date", how="inner")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["Date"], rolling_sharpe(merged["HDE"]),
            label="Full HDE", color="#2563eb", lw=1.5)
    ax.plot(merged["Date"], rolling_sharpe(merged["EqualWt"]),
            label="Equal-weight static ensemble", color="#ef4444", lw=1.5, ls="--")
    ax.plot(merged["Date"], rolling_sharpe(merged["BuyHold"]),
            label="Buy & Hold", color="gray", lw=1.5, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Figure 4.2 — 60-day rolling Sharpe ratio comparison",
                 fontweight="bold")
    ax.set_ylabel("Rolling Sharpe (annualised)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/figure_4_2_rolling_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {EVAL_DIR}/figure_4_2_rolling_sharpe.png")


build_display_items()

# Figure 4.3 - DirAcc against the regime variable with the highest R²
if REGIMES is not None and REGIME_REGRESSION is not None and len(REGIME_REGRESSION):
    top_var = REGIME_REGRESSION.iloc[0]["Variable"]
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(REGIMES[top_var], REGIMES["DirAcc"] * 100, s=100, color="#2563eb")

    for _, r in REGIMES.iterrows():
        ax.annotate(
            pd.Timestamp(r["Window_Start"]).strftime("%Y-%m"),
            (r[top_var], r["DirAcc"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    x = REGIMES[top_var].values
    y = REGIMES["DirAcc"].values * 100

    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(
            xs,
            np.polyval(coef, xs),
            ls="--",
            color="red",
            alpha=0.7,
            label=f"OLS Regression  R²={REGIME_REGRESSION.iloc[0]['R_squared']:.3f}"
        )

    ax.axhline(50, color="gray", lw=0.5, ls=":")
    ax.axhline(53, color="gray", lw=0.5, ls=":", alpha=0.5)

    ax.set_title(
        "Walk-forward Directional Accuracy versus Mean VIX",
        fontweight="bold"
    )
    ax.set_xlabel("Mean VIX")
    ax.set_ylabel("Directional Accuracy (%)")

    legend = ax.legend(
        fontsize=9,
        title="Key",
        title_fontsize=10
    )
    legend.get_title().set_fontweight("bold")

    ax.grid(True, alpha=0.3)

    # Pad so labels stay inside the grid
    ax.margins(x=0.08, y=0.08)

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/figure_4_3_regime_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {EVAL_DIR}/figure_4_3_regime_scatter.png")


# Headline numbers in one JSON for the chapter
def write_summary_report():
    print("\n" + "=" * 78)
    print("CHAPTER 4 — SUMMARY REPORT")
    print("=" * 78)

    # safe-getter for nested dicts
    def get(d, path, default=None):
        for k in path:
            if d is None: return default
            d = d.get(k) if isinstance(d, dict) else None
        return d if d is not None else default

    hde_stats = STRATEGIES["e_HDE"]["stats"]
    bh_stats = STRATEGIES["a_BuyHold"]["stats"]
    eq_stats = STRATEGIES["d_EqualWeight"]["stats"]

    print("\n--- Success Criterion 1: Lower MAE than OLS baseline ---")
    hde_row = TABLE_4_1[TABLE_4_1["Model"] == "HDE"].iloc[0]
    ols_row = TABLE_4_1[TABLE_4_1["Model"] == "Linear"].iloc[0]
    dm_p_hde_vs_ols = DM_MATRIX.loc["HDE", "Linear"]
    print(f"  HDE MAE:    {hde_row['MAE']:.5f}  CI [{hde_row['MAE_CI_lo']:.5f}, {hde_row['MAE_CI_hi']:.5f}]")
    print(f"  OLS MAE:    {ols_row['MAE']:.5f}  CI [{ols_row['MAE_CI_lo']:.5f}, {ols_row['MAE_CI_hi']:.5f}]")
    print(f"  DM p-value: {dm_p_hde_vs_ols:.4f}")
    # PASS only if MAE is lower AND DM is significant
    crit1 = "FAIL" if hde_row["MAE"] >= ols_row["MAE"] else ("PASS" if dm_p_hde_vs_ols < 0.05 else "INCONCLUSIVE")
    print(f"  Verdict:    {crit1}")

    print("\n--- Success Criterion 2: DirAcc > 53–55% naive baseline ---")
    print(f"  HDE DirAcc: {hde_row['DirAcc']:.4f}  CI [{hde_row['DA_CI_lo']:.4f}, {hde_row['DA_CI_hi']:.4f}]")
    print(f"  PT vs 0.5:  p = {hde_row['PT_p_vs_0.5']:.4f}")

    print("\n--- Success Criterion 3: Sharpe > Buy & Hold after costs ---")
    hde_bh_key = "HDE_vs_a_BuyHold"
    p_val = LADDER_PVALS.get(hde_bh_key, {}).get("raw", None)
    print(f"  HDE Sharpe:    {hde_stats['sharpe']:.3f}")
    print(f"  BH Sharpe:     {bh_stats['sharpe']:.3f}")
    if p_val is not None:
        print(f"  JKM p-value:   {p_val:.4f}")
    crit3 = "FAIL" if hde_stats['sharpe'] < bh_stats['sharpe'] else "PASS"
    print(f"  Verdict:       {crit3}")

    print("\n--- Section 4.5 Diagnostic: Dynamic weighting mechanism ---")
    print(f"  HDE Sharpe:       {HDE_VS_EQUAL_TEST['sr1']:+.3f}")
    print(f"  EqualWt Sharpe:   {HDE_VS_EQUAL_TEST['sr2']:+.3f}")
    print(f"  JKM p-value:      {HDE_VS_EQUAL_TEST['p_value']:.4f}")
    if HDE_VS_EQUAL_TEST['p_value'] > 0.05:
        verdict = "INERT — dynamic weighting does not improve on uniform"
    elif HDE_VS_EQUAL_TEST['diff'] < 0:
        verdict = "HARMFUL — dynamic weighting underperforms uniform"
    else:
        verdict = "WORKING — dynamic weighting significantly outperforms uniform"
    print(f"  Verdict:          {verdict}")

    print("\n--- Drawdown attribution ---")
    print(TABLE_4_3.to_string(index=False))

    print("\n--- Robustness: per-ticker alpha cross-section ---")
    cs_mean = PER_TICKER_ALPHA["Annualised_Alpha_%"].mean()
    print(f"  Mean per-ticker annualised alpha: {cs_mean:+.2f}%")

    # Plug straight into the chapter's PLACEHOLDER tags
    report = {
        "headline_verdicts": {
            "criterion_1_mae_vs_ols": crit1,
            "criterion_2_directional_accuracy": f"hit={hde_row['DirAcc']:.4f}",
            "criterion_3_sharpe_vs_bh": crit3,
            "dynamic_weighting_diagnosis": verdict,
        },
        "hde_test_stats": hde_stats,
        "bh_test_stats": bh_stats,
        "eq_weight_test_stats": eq_stats,
        "hde_vs_equal_weight_test": HDE_VS_EQUAL_TEST,
        "per_ticker_alpha_mean_pct": cs_mean,
    }
    with open(f"{EVAL_DIR}/chapter_4_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSummary report → {EVAL_DIR}/chapter_4_summary.json")
    print(f"\nAll evaluation artefacts saved to: {EVAL_DIR}/")


write_summary_report()