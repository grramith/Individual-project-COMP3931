# Tables 4.1 and 4.2
# Use the same backtest settings where possible so model comparisons stay fair

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
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


def pesaran_timmermann(pred, actual, null=0.5):
    pred, actual = np.asarray(pred), np.asarray(actual)
    n = len(pred)
    hit = ((pred > 0) == (actual > 0)).astype(int)
    p_hat = hit.mean()
    if null == 0.5:
        py = (pred > 0).mean()
        pa = (actual > 0).mean()
        p_star = py * pa + (1 - py) * (1 - pa)
        var_p_hat = p_star * (1 - p_star) / n
        var_p_star = (((2 * py - 1) ** 2) * pa * (1 - pa) / n +
                      ((2 * pa - 1) ** 2) * py * (1 - py) / n +
                      4 * py * pa * (1 - py) * (1 - pa) / n ** 2)
        denom = np.sqrt(max(var_p_hat - var_p_star, 1e-12))
        z = (p_hat - p_star) / denom
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        return {"hit_rate": float(p_hat), "stat": float(z), "p_value": float(p),
                "test": "PT-1992"}
    else:
        successes = int(hit.sum())
        result = sp_stats.binomtest(successes, n, p=null, alternative="greater")
        return {"hit_rate": float(p_hat), "stat": float(successes),
                "p_value": float(result.pvalue), "test": f"Binomial>{null}"}


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


def holm_correction(p_dict, alpha=0.05):
    labels = list(p_dict.keys())
    raw = np.array([p_dict[k] for k in labels])
    order = np.argsort(raw)
    m = len(raw)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * raw[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return {
        labels[i]: {
            "raw": float(raw[i]),
            "adj": float(adj[i]),
            "reject": bool(adj[i] < alpha),
        }
        for i in range(m)
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


# Load shared state from script 01
with open(f"{EVAL_DIR}/_state_phase0.pkl", "rb") as f:
    _phase0 = pickle.load(f)
PREDS = _phase0["PREDS"]
HDE_CONFIG = _phase0["HDE_CONFIG"]


# Strategy builders - each returns a preds_df shaped for run_backtest
def preds_for_model(col_name):
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value", col_name]].copy()
    df.rename(columns={col_name: "Prediction"}, inplace=True)
    return df.dropna(subset=["Prediction"])


def build_buy_and_hold():
    # Always-long benchmark
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].copy()
    df["Prediction"] = 1.0
    return df


def build_momentum_12_1():
    # 12-month return minus the most recent month
    master = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    ret_col = "Return_1d" if "Return_1d" in master.columns else "Target_Return"
    frames = []
    for ticker in PREDS["Ticker"].unique():
        h = master[master["Ticker"] == ticker].sort_values("Date").copy()
        h["ret12"] = (1 + h[ret_col]).rolling(252).apply(np.prod, raw=True) - 1
        h["ret1"] = (1 + h[ret_col]).rolling(21).apply(np.prod, raw=True) - 1
        h["mom_12_1"] = h["ret12"] - h["ret1"]
        frames.append(h[["Date", "Ticker", "mom_12_1"]])
    mom = pd.concat(frames, ignore_index=True)
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].merge(
        mom, on=["Date", "Ticker"], how="left"
    )
    df.rename(columns={"mom_12_1": "Prediction"}, inplace=True)
    # Scale to daily so the threshold rule still works
    df["Prediction"] = df["Prediction"] / 252
    return df.dropna(subset=["Prediction"])


def build_equal_weight_ensemble():
    # Fixed-weight ensemble to compare against the adaptive HDE
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value",
                "Pred_RF", "Pred_GB", "Pred_LSTM"]].copy()
    df["Prediction"] = df[["Pred_RF", "Pred_GB", "Pred_LSTM"]].mean(axis=1)
    return df.dropna(subset=["Prediction"])


def build_table_4_1():
    print("\n" + "=" * 78)
    print("TABLE 4.1 — Predictive Performance (95% block bootstrap CIs)")
    print("=" * 78)

    rows = []
    errors = {}

    for model in ["Linear", "Ridge", "RF", "GB", "LSTM", "HDE"]:
        col = f"Pred_{model}"
        if col not in PREDS.columns or PREDS[col].isna().all():
            continue

        sub = PREDS[["Date", "Ticker", "Actual", col]].dropna().copy()
        sub["Error"] = sub["Actual"] - sub[col]

        pred = sub[col].values
        actual = sub["Actual"].values
        errs = sub["Error"].values

        # Need dated errors for the DM matrix later
        errors[model] = sub[["Date", "Ticker", "Error"]].copy()

        abs_errs = np.abs(errs)
        bl = select_block_length(abs_errs)

        mae_pt, (mae_lo, mae_hi), _ = block_bootstrap(
            abs_errs, np.mean, n_boot=5000, block_len=bl
        )

        # Resample pred and actual together so DirAcc stays paired
        paired = np.column_stack([pred, actual])

        def dir_stat(p):
            return float(np.mean((p[:, 0] > 0) == (p[:, 1] > 0)))

        da_pt, (da_lo, da_hi), _ = block_bootstrap(
            paired, dir_stat, n_boot=5000, block_len=bl
        )

        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        pt_res = pesaran_timmermann(pred, actual, null=0.5)

        rows.append({
            "Model": model,
            "N": len(sub),
            "MAE": mae_pt,
            "MAE_CI_lo": mae_lo,
            "MAE_CI_hi": mae_hi,
            "DirAcc": da_pt,
            "DA_CI_lo": da_lo,
            "DA_CI_hi": da_hi,
            "R2": r2,
            "PT_p_vs_0.5": pt_res["p_value"],
            "block_len": bl,
        })

    table = pd.DataFrame(rows)

    # Match models on (Date, Ticker) before running DM
    models = list(errors.keys())
    dm_p = pd.DataFrame(index=models, columns=models, dtype=float)

    for a in models:
        for b in models:
            if a == b:
                dm_p.loc[a, b] = np.nan
                continue

            merged = errors[a].merge(
                errors[b],
                on=["Date", "Ticker"],
                how="inner",
                suffixes=(f"_{a}", f"_{b}")
            )

            if len(merged) < 10:
                dm_p.loc[a, b] = np.nan
                continue

            _, p = diebold_mariano(
                merged[f"Error_{a}"].values,
                merged[f"Error_{b}"].values,
                loss="abs"
            )
            dm_p.loc[a, b] = p

    # Filter out untraded days before the PT test
    hde = PREDS[["Pred_HDE", "Actual"]].dropna()
    hde_traded = hde[hde["Pred_HDE"].abs() > HDE_CONFIG["threshold"]]

    if len(hde_traded) > 0:
        pt_traded_50 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.5
        )
        pt_traded_53 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.53
        )
        pt_traded_55 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.55
        )
    else:
        pt_traded_50 = pt_traded_53 = pt_traded_55 = None

    print(table.to_string(
        index=False,
        formatters={
            "MAE": "{:.5f}".format,
            "MAE_CI_lo": "{:.5f}".format,
            "MAE_CI_hi": "{:.5f}".format,
            "DirAcc": "{:.4f}".format,
            "DA_CI_lo": "{:.4f}".format,
            "DA_CI_hi": "{:.4f}".format,
            "R2": "{:+.4f}".format,
            "PT_p_vs_0.5": "{:.4f}".format,
        }
    ))

    print("\nDiebold–Mariano pairwise p-values (MAE, HLN-corrected):")
    print(dm_p.round(4).to_string())

    print("\nHDE directional accuracy on TRADED days only:")
    for label, r in [("vs 0.50", pt_traded_50),
                     ("vs 0.53", pt_traded_53),
                     ("vs 0.55", pt_traded_55)]:
        if r is None:
            continue
        print(f"  {label}:  hit={r['hit_rate']:.4f}  p={r['p_value']:.4f}  ({r['test']})")

    table.to_csv(f"{EVAL_DIR}/table_4_1_predictive_performance.csv", index=False)
    dm_p.to_csv(f"{EVAL_DIR}/table_4_1_dm_matrix.csv")
    return table, dm_p, errors


# Strict version - do not silently compare misaligned forecast errors
def diebold_mariano(e1, e2, h=1, loss="abs"):
    e1, e2 = np.asarray(e1), np.asarray(e2)

    if len(e1) != len(e2):
        raise ValueError(
            f"Diebold-Mariano requires aligned error series of equal length, "
            f"got {len(e1)} and {len(e2)}."
        )

    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(loss)

    T = len(d)
    d_bar = np.mean(d)

    gamma_0 = np.var(d, ddof=0)
    gamma = [gamma_0]
    for k in range(1, h):
        gk = np.mean((d[:-k] - d_bar) * (d[k:] - d_bar))
        gamma.append(gk)

    var_d = gamma[0] + 2 * sum(gamma[1:])
    var_d = max(var_d, 1e-12) / T

    dm = d_bar / np.sqrt(var_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln_factor
    p = 2 * (1 - sp_stats.t.cdf(abs(dm_hln), df=T - 1))
    return float(dm_hln), float(p)


TABLE_4_1, DM_MATRIX, FORECAST_ERRORS = build_table_4_1()


# Table 4.2 - baseline ladder
def run_strategy(label, preds_df, **override):
    # Default to the tuned HDE settings, override per-baseline as needed
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


def build_table_4_2():
    print("\n" + "=" * 78)
    print("TABLE 4.2 — Baseline Ladder (95% CIs, paired tests vs HDE)")
    print("=" * 78)

    strategies = {}

    # Buy & hold with overlay flags off
    bh_preds = build_buy_and_hold()
    strategies["a_BuyHold"] = run_strategy(
        "Buy & Hold", bh_preds,
        threshold=0.0, use_threshold=False, use_vix_filter=False,
        use_taper=False, use_fractional=False, allow_short=False,
    )

    # Standard momentum baseline
    try:
        mom_preds = build_momentum_12_1()
        strategies["b_Momentum"] = run_strategy("12-1 Momentum", mom_preds)
    except Exception as e:
        print(f"  [warn] momentum baseline failed: {e}")

    # Check whether simple OLS predictions benefit from the same trading overlay
    if "Pred_Linear" in PREDS.columns and not PREDS["Pred_Linear"].isna().all():
        strategies["c_OLS_overlay"] = run_strategy(
            "OLS + overlay", preds_for_model("Pred_Linear"))

    # Compare HDE with the same models but no adaptive weights
    strategies["d_EqualWeight"] = run_strategy(
        "Equal-weight static ens.", build_equal_weight_ensemble())

    strategies["e_HDE"] = run_strategy("Full HDE", preds_for_model("Pred_HDE"))

    rows = []
    for key, res in strategies.items():
        s = res["stats"]
        rets = res["daily_returns"]
        bl = select_block_length(rets)
        sr_pt, (sr_lo, sr_hi), _ = block_bootstrap(
            rets, lambda x: sharpe_annualised(x), n_boot=5000, block_len=bl)
        rows.append({
            "Strategy": res["label"],
            "Total Return %": round(s["total_return_pct"], 1),
            "Sharpe": round(s["sharpe"], 3),
            "Sharpe_CI_lo": round(sr_lo, 3),
            "Sharpe_CI_hi": round(sr_hi, 3),
            "Sortino": round(s["sortino"], 3),
            "Calmar": round(s["calmar"], 3),
            "Max DD %": round(s["max_drawdown"] * 100, 1),
            "Exposure %": round(s["avg_exposure"] * 100, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Merge on Date so JKM gets paired returns
    print("\nJobson–Korkie–Memmel Sharpe tests (vs Full HDE):")
    hde_rets = strategies["e_HDE"]["daily_returns"]
    pair_pvals = {}
    for key, res in strategies.items():
        if key == "e_HDE":
            continue
        hde_port = strategies["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "hde"})
        other_port = res["portfolio"][["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "other"})
        merged = hde_port.merge(other_port, on="Date", how="inner")
        t = sharpe_difference_test(merged["hde"].values, merged["other"].values)
        print(f"  HDE vs {res['label']:<26}  "
              f"ΔSR={t['diff']:+.3f}  z={t['z']:+.2f}  p={t['p_value']:.4f}")
        pair_pvals[f"HDE_vs_{key}"] = t["p_value"]

    # Adjust the four Sharpe comparisons for multiple testing
    adj = holm_correction(pair_pvals)
    print("\nHolm-corrected p-values:")
    for k, v in adj.items():
        mark = "★" if v["reject"] else " "
        print(f"  {mark} {k:<30}  raw={v['raw']:.4f}  adj={v['adj']:.4f}")

    table.to_csv(f"{EVAL_DIR}/table_4_2_baseline_ladder.csv", index=False)
    pd.DataFrame(adj).T.to_csv(f"{EVAL_DIR}/table_4_2_sharpe_tests.csv")
    return strategies, table, adj


STRATEGIES, TABLE_4_2, LADDER_PVALS = build_table_4_2()


# Save outputs so scripts 04 and 05 can pick them up
with open(f"{EVAL_DIR}/_state_phase3.pkl", "wb") as f:
    pickle.dump({
        "STRATEGIES": STRATEGIES,
        "TABLE_4_1": TABLE_4_1,
        "TABLE_4_2": TABLE_4_2,
        "DM_MATRIX": DM_MATRIX,
        "LADDER_PVALS": LADDER_PVALS,
        "FORECAST_ERRORS": FORECAST_ERRORS,
    }, f)
print(f"saved phase-3 state to {EVAL_DIR}/_state_phase3.pkl")