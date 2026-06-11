"""
Sweep gating uplift of daily_macro_v3 against downstream models (v10 to v19, plus 15min models).
Runs both 'symbol' and 'day' gating modes for LONG and SHORT trades.
Outputs a summary markdown table comparing all models, and generates
individual detailed markdown reports for each downstream model.
Dynamically resolves datasets for each downstream model from their lock files.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.append(os.getcwd())

from scripts.gauntlet.paths import gauntlet_root
from scripts.gauntlet.uplift import find_latest_completed_run, load_preds_and_df, extract_topk_trades

DAILY_MODEL = "daily_macro_v3"
DAILY_DATASET = "data/ranking_data_daily_macro_v3.csv"

MODELS_TO_SWEEP = [
    "v10_native_1h",
    "v10_depth4_1h",
    "v11_utility_1h",
    "v12_lambdamart_1h",
    "v13_ndcg_raw_1h",
    "v14_lambdamart_no_es_1h",
    "v15_lambdamart_es_1h",
    "v16_binary_breakout_1h",
    "v17_random_forest_1h",
    "v18_random_forest_1h",
    "v19_catboost_1h",
    "v2_15min_3y",
    "v3_15min_clean"
]

def run_uplift_for_model(df_daily, downstream_model, gate_mode, downstream_k=3, cost_bps=6.0):
    try:
        downstream_run = find_latest_completed_run(downstream_model)
    except Exception:
        return None
        
    # Dynamically resolve dataset path from downstream lock file
    downstream_dataset = None
    try:
        lock_path = os.path.join(downstream_run, "config.lock.json")
        if os.path.exists(lock_path):
            with open(lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            downstream_dataset = lock_data.get("dataset_path")
    except Exception:
        pass
        
    if not downstream_dataset:
        # Fallback default
        downstream_dataset = "data/ranking_data_upstox_1h_v3_3y.csv"
        
    try:
        df_intra, _ = load_preds_and_df(downstream_run, downstream_dataset)
    except Exception as e:
        print(f"  [ERROR] Failed to load data/preds for {downstream_model}: {e}")
        return None
    
    # Align dates
    df_daily["trade_date"] = df_daily["DateTime"].dt.normalize()
    df_intra["trade_date"] = df_intra["DateTime"].dt.normalize()

    daily_dates = sorted(df_daily["trade_date"].unique())
    intra_dates = sorted(df_intra["trade_date"].unique())

    # Build PIT join date map
    trade_to_daily_map = {}
    for t_date in intra_dates:
        prev_dates = [d for d in daily_dates if d < t_date]
        if prev_dates:
            trade_to_daily_map[t_date] = max(prev_dates)

    overlap_trade_dates = [t for t in intra_dates if t in trade_to_daily_map]
    if not overlap_trade_dates:
        return None
        
    # Resolve downstream label column dynamically from lock dataset specs
    downstream_label_col = "Return"
    try:
        from scripts.gauntlet.cli import REGISTERED_DATASETS
        for ds_name, spec in REGISTERED_DATASETS.items():
            if spec.path == downstream_dataset:
                downstream_label_col = spec.label_col
                break
    except Exception:
        pass
        
    cost_val = cost_bps / 10000.0
    res_out = {
        "downstream_run_id": os.path.basename(downstream_run),
        "overlap_days": len(overlap_trade_dates),
        "start_date": overlap_trade_dates[0].date().isoformat(),
        "end_date": overlap_trade_dates[-1].date().isoformat(),
        "sides": {}
    }

    for side in ["long", "short"]:
        df_topk = extract_topk_trades(df_intra, K=downstream_k, side=side, label_col=downstream_label_col)
        if df_topk.empty:
            continue
            
        df_topk["daily_date"] = df_topk["trade_date"].map(trade_to_daily_map)
        df_topk = df_topk.dropna(subset=["daily_date"])
        
        if gate_mode == "symbol":
            df_daily_score_col = "pred_long" if side == "long" else "pred_short"
            df_daily_subset = df_daily[["DateTime", "Ticker", df_daily_score_col]].copy()
            df_daily_subset = df_daily_subset.rename(columns={
                "DateTime": "daily_date",
                df_daily_score_col: "daily_score"
            })
            df_daily_subset["daily_rank"] = df_daily_subset.groupby("daily_date")["daily_score"].rank(pct=True)
            df_joined = pd.merge(df_topk, df_daily_subset, on=["daily_date", "Ticker"], how="left")
            df_joined = df_joined.dropna(subset=["daily_rank"])
            
            df_joined["gated_category"] = "neutral"
            df_joined.loc[df_joined["daily_rank"] >= 0.70, "gated_category"] = "favorable"
            df_joined.loc[df_joined["daily_rank"] <= 0.30, "gated_category"] = "unfavorable"
        else:
            df_daily_score_col = "pred_long" if side == "long" else "pred_short"
            daily_groups = df_daily.groupby("DateTime")
            daily_sentiment = daily_groups[df_daily_score_col].apply(
                lambda s: s.nlargest(max(1, int(len(s) * 0.10))).mean()
            ).to_dict()
            
            df_topk["daily_sentiment"] = df_topk["daily_date"].map(daily_sentiment)
            df_joined = df_topk.dropna(subset=["daily_sentiment"]).copy()
            
            unique_sentiments = df_joined.drop_duplicates("daily_date")["daily_sentiment"].values
            if len(unique_sentiments) >= 3:
                q33 = np.percentile(unique_sentiments, 33.3)
                q66 = np.percentile(unique_sentiments, 66.6)
            else:
                q33 = q66 = 0.0
                
            df_joined["gated_category"] = "neutral"
            df_joined.loc[df_joined["daily_sentiment"] > q66, "gated_category"] = "favorable"
            df_joined.loc[df_joined["daily_sentiment"] <= q33, "gated_category"] = "unfavorable"

        fav_trades = df_joined[df_joined["gated_category"] == "favorable"]
        unfav_trades = df_joined[df_joined["gated_category"] == "unfavorable"]

        n_fav = len(fav_trades)
        n_unfav = len(unfav_trades)

        if n_fav < 10 or n_unfav < 10:
            continue

        fav_rets = fav_trades["trade_return"].values
        unfav_rets = unfav_trades["trade_return"].values

        fav_mean_net = (fav_rets.mean() - cost_val) * 10000
        unfav_mean_net = (unfav_rets.mean() - cost_val) * 10000
        uplift = fav_mean_net - unfav_mean_net
        
        fav_wr = (fav_rets > 0).mean()
        unfav_wr = (unfav_rets > 0).mean()
        
        t_stat, p_val = ttest_ind(fav_rets - cost_val, unfav_rets - cost_val, equal_var=False)
        
        res_out["sides"][side] = {
            "n_fav": n_fav,
            "n_unfav": n_unfav,
            "fav_net": fav_mean_net,
            "unfav_net": unfav_mean_net,
            "fav_wr": fav_wr,
            "unfav_wr": unfav_wr,
            "uplift": uplift,
            "t_stat": t_stat,
            "p_val": p_val
        }
    return res_out

def write_individual_report(model_name, daily_run_id, model_data):
    report_path = f"finalgo-memory-layer/finalgo/08. Model Analysis/Gauntlet Reports/v3_gating_uplift_{model_name}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🛡️ Gating Uplift Certification Report: `{model_name}`\n\n")
        f.write("## 📌 Metadata\n")
        f.write(f"- **Daily Model**: `daily_macro_v3` (Run ID: `{daily_run_id}`)\n")
        f.write(f"- **Downstream Model**: `{model_name}` (Run ID: `{model_data['run_id']}`)\n")
        f.write(f"- **Overlap OOS Window**: {model_data['overlap_days']} days ({model_data['start_date']} to {model_data['end_date']})\n")
        f.write("- **Friction Cost Level**: 6.0 bps round-trip\n")
        f.write("- **Primary Selection K**: 3 trades per query bar\n\n")
        
        f.write("## 📊 Gating Performance Details\n\n")
        
        for mode in ["symbol", "day"]:
            mode_label = "Symbol Gating (Top 30% Rank)" if mode == "symbol" else "Day Gating (Tercile Market Aggregate)"
            f.write(f"### ⚙️ {mode_label}\n\n")
            
            data = model_data.get(mode)
            if not data or not data["sides"]:
                f.write("*Insufficient trades/run data available for evaluation in this mode.*\n\n")
                continue
                
            for side in ["long", "short"]:
                side_data = data["sides"].get(side)
                if not side_data:
                    continue
                    
                passed = "✅ PASSED" if (side_data["uplift"] >= 2.0 and side_data["t_stat"] >= 2.0) else "❌ FAILED"
                
                f.write(f"#### Downstream {side.upper()} Trades:\n")
                f.write(f"- **Favorable Trades (Gated)**: Count = {side_data['n_fav']} | Net Return = {side_data['fav_net']:+.2f} bps | WR = {side_data['fav_wr']:.1%}\n")
                f.write(f"- **Unfavorable Trades (Blocked)**: Count = {side_data['n_unfav']} | Net Return = {side_data['unfav_net']:+.2f} bps | WR = {side_data['unfav_wr']:.1%}\n")
                f.write(f"- **Performance Uplift**: **`{side_data['uplift']:+.2f} bps`**\n")
                f.write(f"- **T-Statistic**: `{side_data['t_stat']:.2f}` (p-value: `{side_data['p_val']:.4f}`)\n")
                f.write(f"- **Certification Verdict**: **`{passed}`**\n\n")
                
        f.write("## ⚖️ Integration Verdict\n")
        long_pass_sym = model_data.get("symbol", {}).get("sides", {}).get("long", {}).get("uplift", 0) >= 2.0 and model_data.get("symbol", {}).get("sides", {}).get("long", {}).get("t_stat", 0) >= 2.0
        short_pass_sym = model_data.get("symbol", {}).get("sides", {}).get("short", {}).get("uplift", 0) >= 2.0 and model_data.get("symbol", {}).get("sides", {}).get("short", {}).get("t_stat", 0) >= 2.0
        
        if long_pass_sym or short_pass_sym:
            f.write("> [!TIP]\n")
            f.write(f"> **Gating Certification has PASSED for {model_name} in one or more sides!** Integration is technically eligible for production deployment, subject to variance tests.\n")
        else:
            f.write("> [!WARNING]\n")
            f.write(f"> **Gating Certification has FAILED for {model_name} across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.\n")
            
        f.write("\n---\n*Report generated programmatically via evaluate_v3_gating_sweeps.py.*\n")

def main():
    print("=" * 80)
    print("daily_macro_v3 GATING SWEEP PIPELINE (v10 -> v19 + 15M)")
    print("=" * 80)
    
    try:
        daily_run = find_latest_completed_run(DAILY_MODEL)
        daily_run_id = os.path.basename(daily_run)
        print(f"Loading V3 daily predictions from: {daily_run}")
        df_daily, _ = load_preds_and_df(daily_run, DAILY_DATASET)
    except Exception as e:
        print(f"[FATAL] Failed to load daily predictions: {e}")
        sys.exit(1)
        
    combined_results = []
    
    for model in MODELS_TO_SWEEP:
        print(f"Processing model: {model}...")
        
        model_report_data = {
            "run_id": "N/A",
            "overlap_days": 0,
            "start_date": "N/A",
            "end_date": "N/A",
            "symbol": None,
            "day": None
        }
        
        has_data = False
        
        for gate_mode in ["symbol", "day"]:
            res = run_uplift_for_model(df_daily, model, gate_mode)
            if res is None:
                continue
                
            has_data = True
            model_report_data["run_id"] = res["downstream_run_id"]
            model_report_data["overlap_days"] = res["overlap_days"]
            model_report_data["start_date"] = res["start_date"]
            model_report_data["end_date"] = res["end_date"]
            model_report_data[gate_mode] = res
            
            for side in ["long", "short"]:
                if side in res["sides"]:
                    metrics = res["sides"][side]
                    combined_results.append({
                        "Model": model,
                        "Gate Mode": gate_mode,
                        "Side": side.upper(),
                        "Fav Trades": metrics["n_fav"],
                        "Unfav Trades": metrics["n_unfav"],
                        "Fav Net Return (bps)": f"{metrics['fav_net']:+.2f}",
                        "Unfav Net Return (bps)": f"{metrics['unfav_net']:+.2f}",
                        "Net Uplift (bps)": f"{metrics['uplift']:+.2f}",
                        "T-Statistic": f"{metrics['t_stat']:.2f}",
                        "P-Value": f"{metrics['p_val']:.4f}",
                        "Status": "PASSED" if (metrics["uplift"] >= 2.0 and metrics["t_stat"] >= 2.0) else "FAILED"
                    })
                    
        if has_data:
            write_individual_report(model, daily_run_id, model_report_data)
            print(f"  [SUCCESS] Generated individual report for {model}")
        else:
            print(f"  [SKIPPED] No completed runs found for {model}")
            
    df_res = pd.DataFrame(combined_results)
    
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY TABLE")
    print("=" * 80)
    if not df_res.empty:
        print(df_res.to_markdown(index=False))
    else:
        print("No evaluation data generated.")
        
    # Save combined report
    combined_report_path = "finalgo-memory-layer/finalgo/08. Model Analysis/Gauntlet Reports/v3_gating_sweep_report.md"
    os.makedirs(os.path.dirname(combined_report_path), exist_ok=True)
    
    with open(combined_report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 daily_macro_v3 Gating Sweep Report (v10 to v19 + 15M)\n\n")
        f.write(f"- **Evaluated At**: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"- **Daily Model**: `daily_macro_v3` (Run ID: `{daily_run_id}`)\n")
        f.write(f"- **Friction Applied**: 6.0 bps round-trip\n\n")
        f.write("## Summary Sweep Results Table\n\n")
        if not df_res.empty:
            f.write(df_res.to_markdown(index=False))
        else:
            f.write("*No evaluation data generated.*\n")
        f.write("\n\n---\n*Report generated programmatically via evaluate_v3_gating_sweeps.py.*\n")
        
    print(f"\nSaved combined Markdown sweep report to: {combined_report_path}")

if __name__ == "__main__":
    main()
