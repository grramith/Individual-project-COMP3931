def preds_for_model(col_name):
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value", col_name]].copy()
    df.rename(columns={col_name: "Prediction"}, inplace=True)
    return df.dropna(subset=["Prediction"])


def build_buy_and_hold():
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].copy()
    df["Prediction"] = 1.0
    return df


def build_momentum_12_1():
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
    df["Prediction"] = df["Prediction"] / 252
    return df.dropna(subset=["Prediction"])


def build_equal_weight_ensemble():
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value",
                "Pred_RF", "Pred_GB", "Pred_LSTM"]].copy()
    df["Prediction"] = df[["Pred_RF", "Pred_GB", "Pred_LSTM"]].mean(axis=1)
    return df.dropna(subset=["Prediction"])

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


def build_table_4_1():
    print("\n" + "=" * 78)
    print("TABLE 4.1 — Predictive Performance (95% block bootstrap CIs)")
    print("=" * 78)
    rows = []
    errors = {}
    return rows, errors