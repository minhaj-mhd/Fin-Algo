import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import argparse

sys.path.append(os.getcwd())

from scripts.gauntlet.paths import gauntlet_root

def find_latest_completed_run(model_name: str) -> str:
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(f"Gauntlet ledger not found at {ledger_path}")
    latest_run_id = None
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "completed" and record.get("model_name") == model_name:
                latest_run_id = record.get("run_id")
    if latest_run_id is None:
        raise ValueError(f"No completed gauntlet run found for model {model_name}")
    return os.path.join(gauntlet_root(), latest_run_id)

def load_preds_and_df(run_dir: str, dataset_path: str):
    npz_path = os.path.join(run_dir, "preds.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Predictions file not found at {npz_path}")
    
    npz = np.load(npz_path)
    df = pd.read_csv(dataset_path)
    
    # Reconstruct the test set DataFrame aligned with predictions
    test_idx = npz["idx"]
    df_test = df.iloc[test_idx].copy()
    
    if "rl" in npz:
        df_test["pred_long"] = npz["rl"]
    if "rs" in npz:
        df_test["pred_short"] = npz["rs"]
        
    df_test["DateTime"] = pd.to_datetime(df_test["DateTime"])
    if df_test["DateTime"].dt.tz is not None:
        df_test["DateTime"] = df_test["DateTime"].dt.tz_localize(None)
    return df_test, npz

def extract_topk_trades(df_intra, K=3, side="long", label_col="Return"):
    score_col = "pred_long" if side == "long" else "pred_short"
    if score_col not in df_intra.columns:
        return pd.DataFrame()
        
    # Sort within each Query_ID by score descending
    df_sorted = df_intra.sort_values(["Query_ID", score_col], ascending=[True, False])
    
    # Take the top K per Query_ID
    df_topk = df_sorted.groupby("Query_ID").head(K).copy()
    df_topk["trade_return"] = df_topk[label_col] if side == "long" else -df_topk[label_col]
    return df_topk

def bootstrap_uplift_ci(fav_rets, unfav_rets, cost, n_reps=1000, seed=42):
    rng = np.random.default_rng(seed)
    uplifts = []
    
    fav_rets_net = fav_rets - cost
    unfav_rets_net = unfav_rets - cost
    
    for _ in range(n_reps):
        fav_sample = rng.choice(fav_rets_net, size=len(fav_rets_net), replace=True)
        unfav_sample = rng.choice(unfav_rets_net, size=len(unfav_rets_net), replace=True)
        uplifts.append((fav_sample.mean() - unfav_sample.mean()) * 10000) # In bps
        
    return float(np.percentile(uplifts, 2.5)), float(np.percentile(uplifts, 97.5))

def main():
    parser = argparse.ArgumentParser(description="Gatekeeper-Uplift Certification Engine (D5)")
    parser.add_argument("--daily-model", type=str, default="daily_macro_v2", help="Daily model name")
    parser.add_argument("--downstream-model", type=str, default="v8_upstox_3y", help="Downstream model name")
    parser.add_argument("--daily-dataset", type=str, default="data/ranking_data_daily_macro_v2.csv", help="Daily dataset path")
    parser.add_argument("--downstream-dataset", type=str, default="data/ranking_data_upstox_1h_v3_3y.csv", help="Downstream dataset path")
    parser.add_argument("--gate-mode", type=str, choices=["day", "symbol"], default="day", help="Gate alignment mode")
    parser.add_argument("--downstream-k", type=int, default=3, help="K for downstream model selections")
    parser.add_argument("--cost-bps", type=float, default=6.0, help="Downstream cost in bps")
    args = parser.parse_args()

    print("=" * 70)
    print("GATEKEEPER UPLIFT CERTIFICATION ENGINE")
    print(f"Daily Model     : {args.daily_model}")
    print(f"Downstream Model: {args.downstream_model}")
    print(f"Gate Mode       : {args.gate_mode}")
    print("=" * 70)

    try:
        daily_run = find_latest_completed_run(args.daily_model)
        downstream_run = find_latest_completed_run(args.downstream_model)
        print(f"Found Daily run     : {daily_run}")
        print(f"Found Downstream run: {downstream_run}")
    except Exception as e:
        print(f"[FATAL] Failed to locate run directories: {e}")
        sys.exit(1)

    print("\nLoading datasets and predictions...")
    # Read daily predictions
    df_daily, daily_npz = load_preds_and_df(daily_run, args.daily_dataset)
    # Read downstream predictions
    df_intra, intra_npz = load_preds_and_df(downstream_run, args.downstream_dataset)

    print(f"Loaded Daily predictions: {len(df_daily):,} rows")
    print(f"Loaded Downstream predictions: {len(df_intra):,} rows")

    # Align dates
    df_daily["trade_date"] = df_daily["DateTime"].dt.normalize()
    df_intra["trade_date"] = df_intra["DateTime"].dt.normalize()

    daily_dates = sorted(df_daily["trade_date"].unique())
    intra_dates = sorted(df_intra["trade_date"].unique())

    # Build trade date to previous daily date mapping (PIT join)
    trade_to_daily_map = {}
    for t_date in intra_dates:
        prev_dates = [d for d in daily_dates if d < t_date]
        if prev_dates:
            trade_to_daily_map[t_date] = max(prev_dates)

    overlap_trade_dates = [t for t in intra_dates if t in trade_to_daily_map]
    print(f"Overlap OOS trading window: {len(overlap_trade_dates)} days ({overlap_trade_dates[0].date()} to {overlap_trade_dates[-1].date()})")

    # Resolve daily label column dynamically
    daily_label_col = "Label_3D"
    try:
        daily_lock_path = os.path.join(daily_run, "config.lock.json")
        if os.path.exists(daily_lock_path):
            with open(daily_lock_path, "r", encoding="utf-8") as f:
                daily_lock_data = json.load(f)
            daily_ds_path = daily_lock_data.get("dataset_path")
            if daily_ds_path:
                from scripts.gauntlet.cli import REGISTERED_DATASETS
                for ds_name, spec in REGISTERED_DATASETS.items():
                    if spec.path == daily_ds_path:
                        daily_label_col = spec.label_col
                        print(f"Dynamically resolved daily label column: '{daily_label_col}'")
                        break
    except Exception as e:
        print(f"[WARNING] Failed to dynamically resolve daily label column: {e}")

    # Resolve downstream label column dynamically
    downstream_label_col = "Return"  # Default fallback
    try:
        downstream_lock_path = os.path.join(downstream_run, "config.lock.json")
        if os.path.exists(downstream_lock_path):
            with open(downstream_lock_path, "r", encoding="utf-8") as f:
                downstream_lock_data = json.load(f)
            downstream_ds_path = downstream_lock_data.get("dataset_path")
            if downstream_ds_path:
                from scripts.gauntlet.cli import REGISTERED_DATASETS
                for ds_name, spec in REGISTERED_DATASETS.items():
                    if spec.path == downstream_ds_path:
                        downstream_label_col = spec.label_col
                        print(f"Dynamically resolved downstream label column: '{downstream_label_col}'")
                        break
    except Exception as e:
        print(f"[WARNING] Failed to dynamically resolve downstream label column: {e}")

    results = {}
    cost_val = args.cost_bps / 10000.0

    for side in ["long", "short"]:
        print(f"\n--- Gating Downstream {side.upper()} Trades (K={args.downstream_k}) ---")
        
        # 1. Get top-K downstream trades
        df_topk = extract_topk_trades(df_intra, K=args.downstream_k, side=side, label_col=downstream_label_col)
        if df_topk.empty:
            print(f"No downstream trades for side {side}")
            continue

        # Map to previous daily dates
        df_topk["daily_date"] = df_topk["trade_date"].map(trade_to_daily_map)
        df_topk = df_topk.dropna(subset=["daily_date"])

        # 2. Join Daily predictions
        if args.gate_mode == "symbol":
            # Symbol-level ranking gate
            # Calculate daily ranks per ticker within each daily date
            score_col = "daily_score_long" if side == "long" else "daily_score_short"
            df_daily_score_col = "pred_long" if side == "long" else "pred_short"
            
            df_daily_subset = df_daily[["DateTime", "Ticker", df_daily_score_col]].copy()
            df_daily_subset = df_daily_subset.rename(columns={
                "DateTime": "daily_date",
                df_daily_score_col: "daily_score"
            })
            
            # Rank percentiles
            df_daily_subset["daily_rank"] = df_daily_subset.groupby("daily_date")["daily_score"].rank(pct=True)
            
            # Merge
            df_joined = pd.merge(df_topk, df_daily_subset, on=["daily_date", "Ticker"], how="left")
            df_joined = df_joined.dropna(subset=["daily_rank"])
            
            # Gating categories
            df_joined["gated_category"] = "neutral"
            df_joined.loc[df_joined["daily_rank"] >= 0.70, "gated_category"] = "favorable"
            df_joined.loc[df_joined["daily_rank"] <= 0.30, "gated_category"] = "unfavorable"
            
        else:
            # Day-level market aggregate gate
            # Daily sentiment = mean of top decile scores on that day
            df_daily_score_col = "pred_long" if side == "long" else "pred_short"
            
            daily_groups = df_daily.groupby("DateTime")
            daily_sentiment = daily_groups[df_daily_score_col].apply(
                lambda s: s.nlargest(max(1, int(len(s) * 0.10))).mean()
            ).to_dict()
            
            df_topk["daily_sentiment"] = df_topk["daily_date"].map(daily_sentiment)
            df_joined = df_topk.dropna(subset=["daily_sentiment"]).copy()
            
            # Tercile thresholds based on unique daily sentiment values
            unique_sentiments = df_joined.drop_duplicates("daily_date")["daily_sentiment"].values
            if len(unique_sentiments) >= 3:
                q33 = np.percentile(unique_sentiments, 33.3)
                q66 = np.percentile(unique_sentiments, 66.6)
            else:
                q33 = q66 = 0.0
                
            df_joined["gated_category"] = "neutral"
            df_joined.loc[df_joined["daily_sentiment"] > q66, "gated_category"] = "favorable"
            df_joined.loc[df_joined["daily_sentiment"] <= q33, "gated_category"] = "unfavorable"

        # Filter categories
        fav_trades = df_joined[df_joined["gated_category"] == "favorable"]
        unfav_trades = df_joined[df_joined["gated_category"] == "unfavorable"]

        n_fav = len(fav_trades)
        n_unfav = len(unfav_trades)

        if n_fav < 10 or n_unfav < 10:
            print(f"  [WARN] Insufficient samples for comparison: Fav={n_fav}, Unfav={n_unfav}")
            continue

        fav_rets = fav_trades["trade_return"].values
        unfav_rets = unfav_trades["trade_return"].values

        fav_mean_net = (fav_rets.mean() - cost_val) * 10000
        unfav_mean_net = (unfav_rets.mean() - cost_val) * 10000
        
        uplift = fav_mean_net - unfav_mean_net
        
        fav_wr = (fav_rets > 0).mean()
        unfav_wr = (unfav_rets > 0).mean()

        # Two-sample t-test
        t_stat, p_val = ttest_ind(fav_rets - cost_val, unfav_rets - cost_val, equal_var=False)

        # Bootstrap confidence interval
        ci_lower, ci_upper = bootstrap_uplift_ci(fav_rets, unfav_rets, cost_val)

        print(f"  Gated Trades Count: Fav={n_fav} | Unfav={n_unfav}")
        print(f"  Favorable Trades  : Net Return={fav_mean_net:+.2f} bps | WR={fav_wr:.1%}")
        print(f"  Unfavorable Trades: Net Return={unfav_mean_net:+.2f} bps | WR={unfav_wr:.1%}")
        print(f"  Performance Uplift: {uplift:+.2f} bps (95% CI: [{ci_lower:+.2f}, {ci_upper:+.2f}] bps)")
        print(f"  T-statistic       : {t_stat:.2f} (p-value: {p_val:.4f})")

        passed = uplift >= 2.0 and t_stat >= 2.0
        print(f"  Uplift Certification Status: {'[PASSED]' if passed else '[FAILED]'}")

        results[side] = {
            "n_favorable": int(n_fav),
            "n_unfavorable": int(n_unfav),
            "favorable_net_bps": float(fav_mean_net),
            "unfavorable_net_bps": float(unfav_mean_net),
            "favorable_wr": float(fav_wr),
            "unfavorable_wr": float(unfav_wr),
            "uplift_bps": float(uplift),
            "uplift_t_stat": float(t_stat),
            "uplift_p_value": float(p_val),
            "ci_lower_bps": float(ci_lower),
            "ci_upper_bps": float(ci_upper),
            "certified": bool(passed)
        }

    # Save output report in daily model run directory
    out_report_path = os.path.join(daily_run, "gatekeeper_uplift_report.json")
    report_data = {
        "daily_model": args.daily_model,
        "daily_run_id": os.path.basename(daily_run),
        "downstream_model": args.downstream_model,
        "downstream_run_id": os.path.basename(downstream_run),
        "gate_mode": args.gate_mode,
        "downstream_k": args.downstream_k,
        "cost_bps": args.cost_bps,
        "results": results,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\nUplift report saved to: {out_report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
