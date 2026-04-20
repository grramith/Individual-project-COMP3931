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

def build_table_4_2():
    print("\n" + "=" * 78)
    print("TABLE 4.2 — Baseline Ladder (95% CIs, paired tests vs HDE)")
    print("=" * 78)

    strategies = {}

    bh_preds = build_buy_and_hold()
    strategies["a_BuyHold"] = run_strategy(
        "Buy & Hold", bh_preds,
        threshold=0.0, use_threshold=False, use_vix_filter=False,
        use_taper=False, use_fractional=False, allow_short=False,
    )

    try:
        mom_preds = build_momentum_12_1()
        strategies["b_Momentum"] = run_strategy("12-1 Momentum", mom_preds)
    except Exception as e:
        print(f"  [warn] momentum baseline failed: {e}")

    if "Pred_Linear" in PREDS.columns and not PREDS["Pred_Linear"].isna().all():
        strategies["c_OLS_overlay"] = run_strategy(
            "OLS + overlay", preds_for_model("Pred_Linear"))

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

    print("\nJobson–Korkie–Memmel Sharpe tests (vs Full HDE):")
    pair_pvals = {}
    for key, res in strategies.items():
        if key == "e_HDE":
            continue
        hde_port = strategies["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(columns={"Strategy_Ret": "hde"})
        other_port = res["portfolio"][["Date", "Strategy_Ret"]].rename(columns={"Strategy_Ret": "other"})
        merged = hde_port.merge(other_port, on="Date", how="inner")
        t = sharpe_difference_test(merged["hde"].values, merged["other"].values)
        print(f"  HDE vs {res['label']:<26}  ΔSR={t['diff']:+.3f}  z={t['z']:+.2f}  p={t['p_value']:.4f}")
        pair_pvals[f"HDE_vs_{key}"] = t["p_value"]

    adj = holm_correction(pair_pvals)
    print("\nHolm-corrected p-values:")
    for k, v in adj.items():
        mark = "★" if v["reject"] else " "
        print(f"  {mark} {k:<30}  raw={v['raw']:.4f}  adj={v['adj']:.4f}")

    table.to_csv(f"{EVAL_DIR}/table_4_2_baseline_ladder.csv", index=False)
    pd.DataFrame(adj).T.to_csv(f"{EVAL_DIR}/table_4_2_sharpe_tests.csv")
    return strategies, table, adj

STRATEGIES, TABLE_4_2, LADDER_PVALS = build_table_4_2()