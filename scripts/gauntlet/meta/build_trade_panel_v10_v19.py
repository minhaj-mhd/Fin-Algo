"""
MV2-R0: Coverage-Rectified Trade Panel Builder for v10 to v19
============================================================
Builds the candidate panel by:
- Loading step_months=2 OOS predictions for models v10 to v19.
- Extracting candidate trades (top-3 per query, long and short) for all models.
- Deduplicating candidate trades by Query_ID, ticker, side.
- Adding scores, z-scores, and percentiles for all 12 models as features.
- Adding daily macro context features (daily_v2, daily_v3).
- Imputing missing standard features using training fold statistics.
- Asserting G1: DEV >= 12 distinct months and >= 5,000 trades.
- Outputting to data/gauntlet/meta/mv_v10_v19/trade_panel.parquet
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.getcwd())

from scripts.gauntlet.paths import gauntlet_root
from scripts.gauntlet.uplift import find_latest_completed_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEV_CUTOFF = pd.Timestamp("2025-01-01")
COST_RATE = 0.0010          # 10 bps
OUT_DIR = os.path.join("data", "gauntlet", "meta", "mv_v10_v19")

MODELS_1H = [
    "v10_native_1h",
    "v10_depth4_1h",
    "v11_utility_1h",
    "v12_lambdamart_1h",
    "v13_ndcg_raw_1h",
    "v14_lambdamart_no_es_1h",
    "v15_lambdamart_es_1h",
    "v16_binary_breakout_1h",
    "v19_catboost_1h",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_run_id_in_ledger(run_dir: str) -> None:
    """
    G6 Guardrail: Verifies that the run_id exists as a completed event
    in the central gauntlet ledger.
    """
    run_id = os.path.basename(os.path.abspath(run_dir))
    ledger_path = os.path.join(gauntlet_root(), "ledger.jsonl")
    if not os.path.exists(ledger_path):
        raise RuntimeError(f"[G6 VIOLATED] Central gauntlet ledger not found at {ledger_path}")
    
    found = False
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("run_id") == run_id:
                    if record.get("event") == "completed" or (record.get("event") is None and "verdicts" in record):
                        # Ensure config hash matches the step_months=2 hash (starts with d795438c)
                        cfg_hash = record.get("config_hash", "")
                        model_name = record.get("model_name", "")
                        if not cfg_hash.startswith("d795438c") and not model_name.startswith("daily_"):
                            raise RuntimeError(
                                f"[G6 VIOLATED] Run ID '{run_id}' has config_hash '{cfg_hash}' which does not match step_months=2. "
                                "We require contiguous step_months=2 runs for the Meta-Veto stacking panel."
                            )
                        found = True
                        break
            except Exception as e:
                if "G6 VIOLATED" in str(e):
                    raise e
                
    if not found:
        raise RuntimeError(
            f"[G6 VIOLATED] Run ID '{run_id}' (from path {run_dir}) was not found as a completed run in the central ledger.\n"
            "This run is invalid or lookahead-contaminated. Panel building aborted."
        )


def load_preds_aligned(run_dir: str, dataset_path: str, usecols: list, model_name: str) -> pd.DataFrame:
    """Load preds.npz and return aligned with matching CSV rows."""
    verify_run_id_in_ledger(run_dir)
    npz_path = os.path.join(run_dir, "preds.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"preds.npz not found: {npz_path}")
    npz = np.load(npz_path)
    df_raw = pd.read_csv(dataset_path, usecols=usecols)
    test_idx = npz["idx"]
    df = df_raw.iloc[test_idx].copy()
    
    # Store predictions
    if "rl" in npz:
        df[f"{model_name}_pred_long"] = npz["rl"]
    if "rs" in npz:
        df[f"{model_name}_pred_short"] = npz["rs"]
        
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    if df["DateTime"].dt.tz is not None:
        df["DateTime"] = df["DateTime"].dt.tz_localize(None)
        
    return df, test_idx


def add_query_stats(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    """Z-score and percentile rank within each Query_ID."""
    if score_col not in df.columns:
        return df
    qmean = df.groupby("Query_ID")[score_col].transform("mean")
    qstd  = df.groupby("Query_ID")[score_col].transform("std").fillna(1e-8)
    df[f"{prefix}_z"]   = (df[score_col] - qmean) / qstd
    df[f"{prefix}_pct"] = df.groupby("Query_ID")[score_col].rank(pct=True)
    return df


def add_day_sentiment(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    """Top-10% average score per calendar day."""
    if score_col not in df.columns:
        return df
    daily = df.groupby("DateTime")[score_col].apply(
        lambda s: s.nlargest(max(1, int(len(s) * 0.10))).mean()
    ).to_dict()
    df[f"{prefix}_sent"] = df["DateTime"].map(daily)
    return df


def extract_topk_trades(df: pd.DataFrame, K: int, side: str, return_col: str, model_name: str) -> pd.DataFrame:
    """Return top-K rows per Query_ID sorted by the model score."""
    score_col = f"{model_name}_pred_long" if side == "long" else f"{model_name}_pred_short"
    if score_col not in df.columns:
        return pd.DataFrame()
    df_sorted = df.sort_values(["Query_ID", score_col], ascending=[True, False])
    df_topk   = df_sorted.groupby("Query_ID").head(K).copy()
    df_topk["trade_return"] = df_topk[return_col] if side == "long" else -df_topk[return_col]
    return df_topk


def build_coverage_matrix(sources: dict) -> pd.DataFrame:
    """Build a month × source coverage table."""
    all_months = set()
    for df in sources.values():
        months = df["DateTime"].dt.to_period("M").unique()
        all_months.update(months)

    months_sorted = sorted(all_months)
    rows = []
    for month in months_sorted:
        row = {"month": str(month)}
        for src_name, df in sources.items():
            mask = df["DateTime"].dt.to_period("M") == month
            row[src_name] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _check_g1(dev_df: pd.DataFrame, coverage_df: pd.DataFrame) -> None:
    dev_months = dev_df["datetime"].dt.to_period("M").nunique()
    dev_trades = len(dev_df)
    if dev_months < 12 or dev_trades < 5_000:
        cov_str = coverage_df.to_string(index=False)
        raise RuntimeError(
            f"\n[G1 VIOLATED] DEV span has {dev_months} months and "
            f"{dev_trades:,} trades — need >= 12 months AND >= 5,000 trades.\n"
            f"Coverage matrix:\n{cov_str}\n"
        )
    print(f"[G1 OK] DEV span: {dev_months} months, {dev_trades:,} trades.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MV2-R0: BUILDING TRADE PANEL FOR v10 TO v19 MODELS")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Locate run directories
    print("\n[1/9] Locating Gauntlet run directories...")
    run_dirs = {}
    for m in MODELS_1H:
        try:
            run_dirs[m] = find_latest_completed_run(m)
            print(f"  {m:<25}: {run_dirs[m]}")
        except Exception as e:
            print(f"[FATAL] Cannot locate run directory for model {m}: {e}")
            sys.exit(1)
            
    try:
        run_daily_v2 = find_latest_completed_run("daily_macro_v2")
        run_daily_v3 = find_latest_completed_run("daily_macro_v3")
        print(f"  daily_macro_v2           : {run_daily_v2}")
        print(f"  daily_macro_v3           : {run_daily_v3}")
    except Exception as e:
        print(f"[FATAL] Cannot locate daily run directories: {e}")
        sys.exit(1)

    # 2. Load and align 1H predictions
    print("\n[2/9] Loading 1H model predictions and verifying index alignment...")
    COLS_1H  = ["DateTime", "Ticker", "Query_ID", "Next_Hour_Return", "Hour", "DayOfWeek"]
    
    # Load first model to establish base test indices
    m0 = MODELS_1H[0]
    df_m0, idx_m0 = load_preds_aligned(run_dirs[m0], "data/ranking_data_upstox_1h_v3_3y.csv", COLS_1H, m0)
    
    # Pre-align and join other models
    model_dfs = {m0: df_m0}
    for m in MODELS_1H[1:]:
        df_m, idx_m = load_preds_aligned(run_dirs[m], "data/ranking_data_upstox_1h_v3_3y.csv", COLS_1H, m)
        if not np.array_equal(idx_m0, idx_m):
            raise RuntimeError(f"[ERROR] Index mismatch between {m0} and {m}. Cannot align predictions.")
        model_dfs[m] = df_m

    # Compute query stats for each model
    print("  Computing query stats (z-score, pct rank) for all models...")
    for m in MODELS_1H:
        df = model_dfs[m]
        df = add_query_stats(df, f"{m}_pred_long", f"{m}_long")
        df = add_query_stats(df, f"{m}_pred_short", f"{m}_short")
        model_dfs[m] = df

    # 3. Load daily macro context predictions
    print("\n[3/9] Loading daily macro context predictions...")
    COLS_DV2 = ["DateTime", "Ticker", "Query_ID", "Nifty50_Dist_SMA_200", "VIX_Level", "VIX_Percentile_1Y"]

    df_dv2 = load_preds_aligned(run_daily_v2, "data/ranking_data_daily_macro_v2.csv", COLS_DV2, "daily_v2")
    # Clean column names to prevent duplication
    df_dv2 = df_dv2[0].rename(columns={"daily_v2_pred_long": "pred_long", "daily_v2_pred_short": "pred_short"})
    
    df_dv3 = load_preds_aligned(run_daily_v3, "data/ranking_data_daily_macro_v3.csv", ["DateTime", "Ticker", "Query_ID"], "daily_v3")
    df_dv3 = df_dv3[0].rename(columns={"daily_v3_pred_long": "pred_long", "daily_v3_pred_short": "pred_short"})

    df_dv2 = add_query_stats(df_dv2, "pred_long",  "daily_v2_long")
    df_dv2 = add_query_stats(df_dv2, "pred_short", "daily_v2_short")
    df_dv2 = add_day_sentiment(df_dv2, "pred_long",  "daily_v2_long")
    df_dv2 = add_day_sentiment(df_dv2, "pred_short", "daily_v2_short")

    df_dv3 = add_query_stats(df_dv3, "pred_long",  "daily_v3_long")
    df_dv3 = add_query_stats(df_dv3, "pred_short", "daily_v3_short")
    df_dv3 = add_day_sentiment(df_dv3, "pred_long",  "daily_v3_long")
    df_dv3 = add_day_sentiment(df_dv3, "pred_short", "daily_v3_short")

    # 4. Coverage matrix
    print("\n[4/9] Building month × source coverage matrix...")
    coverage_sources = {
        "1H_Base": df_m0[["DateTime"]],
        "daily_v2": df_dv2[["DateTime"]],
        "daily_v3": df_dv3[["DateTime"]],
    }
    cov_df = build_coverage_matrix(coverage_sources)
    cov_path = os.path.join(OUT_DIR, "coverage_matrix.csv")
    cov_df.to_csv(cov_path, index=False)
    print("\n  MONTH × SOURCE COVERAGE (row counts)")
    print("  " + cov_df.to_string(index=False).replace("\n", "\n  "))

    # 5. Extract candidate trades (Top-3 per query per model+side)
    print("\n[5/9] Extracting Top-3 candidate trades per query across all models...")
    trade_list = []
    for m in MODELS_1H:
        df_m = model_dfs[m]
        # Long trades
        topk_l = extract_topk_trades(df_m, K=3, side="long", return_col="Next_Hour_Return", model_name=m)
        for _, row in topk_l.iterrows():
            trade_list.append({
                "Query_ID": row["Query_ID"],
                "ticker": row["Ticker"],
                "side": "long",
                "datetime": row["DateTime"],
                "trade_return": row["trade_return"],
                "proposed_by": m,
                "hour": row["Hour"],
                "day_of_week": row["DayOfWeek"]
            })
        # Short trades
        topk_s = extract_topk_trades(df_m, K=3, side="short", return_col="Next_Hour_Return", model_name=m)
        for _, row in topk_s.iterrows():
            trade_list.append({
                "Query_ID": row["Query_ID"],
                "ticker": row["Ticker"],
                "side": "short",
                "datetime": row["DateTime"],
                "trade_return": row["trade_return"],
                "proposed_by": m,
                "hour": row["Hour"],
                "day_of_week": row["DayOfWeek"]
            })

    raw_trades_df = pd.DataFrame(trade_list)
    print(f"  Extracted {len(raw_trades_df):,} raw model-trade pairs.")

    # Deduplicate trades
    print("  Deduplicating candidate trades...")
    panel_df = raw_trades_df.drop_duplicates(subset=["Query_ID", "ticker", "side"]).copy()
    print(f"  Deduplicated to {len(panel_df):,} unique candidate trades.")

    # Add one-hot flags for which models proposed each trade
    print("  Building sibling proposing flags...")
    for m in MODELS_1H:
        proposing_keys = set(zip(
            raw_trades_df[raw_trades_df["proposed_by"] == m]["Query_ID"],
            raw_trades_df[raw_trades_df["proposed_by"] == m]["ticker"],
            raw_trades_df[raw_trades_df["proposed_by"] == m]["side"]
        ))
        panel_df[f"proposed_by_{m}"] = [
            1 if k in proposing_keys else 0 
            for k in zip(panel_df["Query_ID"], panel_df["ticker"], panel_df["side"])
        ]

    # 6. Map predictions of all models as features
    print("\n[6/9] Joining features from all 1H models...")
    # Map the primary model scores (backward compatible names)
    # Let's map properly by joining
    for m in MODELS_1H:
        print(f"  Mapping {m} scores...")
        df_sub = model_dfs[m][["Query_ID", "Ticker", f"{m}_pred_long", f"{m}_pred_short", f"{m}_long_z", f"{m}_long_pct", f"{m}_short_z", f"{m}_short_pct"]].copy()
        
        # Merge
        panel_df = pd.merge(
            panel_df, df_sub,
            left_on=["Query_ID", "ticker"],
            right_on=["Query_ID", "Ticker"],
            how="left"
        ).drop(columns=["Ticker"], errors="ignore")
        
        # Construct features based on side
        panel_df[f"{m}_score"] = np.where(
            panel_df["side"] == "long",
            panel_df[f"{m}_pred_long"], panel_df[f"{m}_pred_short"]
        )
        panel_df[f"{m}_z"] = np.where(
            panel_df["side"] == "long",
            panel_df[f"{m}_long_z"], panel_df[f"{m}_short_z"]
        )
        panel_df[f"{m}_pct"] = np.where(
            panel_df["side"] == "long",
            panel_df[f"{m}_long_pct"], panel_df[f"{m}_short_pct"]
        )
        
        # Drop raw helper columns
        panel_df = panel_df.drop(
            columns=[f"{m}_pred_long", f"{m}_pred_short", f"{m}_long_z", f"{m}_long_pct", f"{m}_short_z", f"{m}_short_pct"],
            errors="ignore"
        )

    # Set anchor compat columns
    anchor = "v10_native_1h"
    panel_df["own_score"] = panel_df[f"{anchor}_score"]
    panel_df["own_z"] = panel_df[f"{anchor}_z"]
    panel_df["own_pct"] = panel_df[f"{anchor}_pct"]
    panel_df["side_is_long"] = (panel_df["side"] == "long").astype(int)

    # 7. Daily macro context join (T-1 PIT)
    print("\n[7/9] Joining daily macro context (T-1 PIT)...")
    panel_df["_trade_date"] = panel_df["datetime"].dt.normalize()
    unique_trade_dates = sorted(panel_df["_trade_date"].unique())
    unique_daily_dates = sorted(df_dv2["DateTime"].dt.normalize().unique())

    trade_date_to_daily = {}
    for td in unique_trade_dates:
        prev = [d for d in unique_daily_dates if d < td]
        if prev:
            trade_date_to_daily[td] = max(prev)

    panel_df["_mapped_daily"] = panel_df["_trade_date"].map(trade_date_to_daily)

    # Merge Daily v2
    dv2_sub = df_dv2[[
        "DateTime", "Ticker",
        "daily_v2_long_pct", "daily_v2_short_pct",
        "daily_v2_long_sent", "daily_v2_short_sent",
        "Nifty50_Dist_SMA_200", "VIX_Level", "VIX_Percentile_1Y"
    ]].copy()
    panel_df = pd.merge(
        panel_df, dv2_sub,
        left_on=["_mapped_daily", "ticker"],
        right_on=["DateTime", "Ticker"],
        how="left"
    ).drop(columns=["DateTime", "Ticker"], errors="ignore")

    panel_df["daily_v2_pct"]  = np.where(
        panel_df["side"] == "long",
        panel_df["daily_v2_long_pct"], panel_df["daily_v2_short_pct"]
    )
    panel_df["daily_v2_sent"] = np.where(
        panel_df["side"] == "long",
        panel_df["daily_v2_long_sent"], panel_df["daily_v2_short_sent"]
    )
    panel_df = panel_df.drop(
        columns=["daily_v2_long_pct", "daily_v2_short_pct",
                 "daily_v2_long_sent", "daily_v2_short_sent"],
        errors="ignore"
    )
    panel_df["macro_gate"] = (panel_df["Nifty50_Dist_SMA_200"] > 0).astype(float)
    panel_df = panel_df.rename(columns={
        "VIX_Level": "vix_level",
        "VIX_Percentile_1Y": "vix_pct"
    })
    panel_df = panel_df.drop(columns=["Nifty50_Dist_SMA_200"], errors="ignore")

    # Merge Daily v3
    dv3_sub = df_dv3[[
        "DateTime", "Ticker",
        "daily_v3_long_pct", "daily_v3_short_pct",
        "daily_v3_long_sent", "daily_v3_short_sent"
    ]].copy()
    panel_df = pd.merge(
        panel_df, dv3_sub,
        left_on=["_mapped_daily", "ticker"],
        right_on=["DateTime", "Ticker"],
        how="left"
    ).drop(columns=["DateTime", "Ticker"], errors="ignore")

    panel_df["daily_v3_pct"]  = np.where(
        panel_df["side"] == "long",
        panel_df["daily_v3_long_pct"], panel_df["daily_v3_short_pct"]
    )
    panel_df["daily_v3_sent"] = np.where(
        panel_df["side"] == "long",
        panel_df["daily_v3_long_sent"], panel_df["daily_v3_short_sent"]
    )
    panel_df = panel_df.drop(
        columns=["daily_v3_long_pct", "daily_v3_short_pct",
                 "daily_v3_long_sent", "daily_v3_short_sent"],
        errors="ignore"
    )
    panel_df = panel_df.drop(columns=["_trade_date", "_mapped_daily"], errors="ignore")

    # 8. Apply tiered feature policy
    print("\n[8/9] Applying tiered feature policy...")
    CORE_COLS = ["own_score", "own_z", "own_pct", "trade_return"]
    before = len(panel_df)
    panel_df = panel_df.dropna(subset=CORE_COLS)
    after = len(panel_df)
    if before > after:
        print(f"  Dropped {before - after:,} rows missing core features.")

    STANDARD_COLS = [
        "daily_v2_pct", "daily_v2_sent",
        "daily_v3_pct", "daily_v3_sent",
        "vix_level", "vix_pct", "macro_gate",
    ]
    for col in STANDARD_COLS:
        if col in panel_df.columns:
            missing_rate = panel_df[col].isna().mean()
            panel_df[f"{col}_missing"] = panel_df[col].isna().astype(int)
            print(f"  {col:<22}: {missing_rate:.1%} missing -> indicator added")
        else:
            print(f"  {col:<22}: column absent — all-1 missing indicator added")
            panel_df[col] = np.nan
            panel_df[f"{col}_missing"] = 1

    # Add binary target and span
    panel_df["y"] = ((panel_df["trade_return"] - COST_RATE) > 0).astype(int)
    panel_df["span"] = np.where(panel_df["datetime"] < DEV_CUTOFF, "DEV", "VAULT")

    dev_df   = panel_df[panel_df["span"] == "DEV"]
    vault_df = panel_df[panel_df["span"] == "VAULT"]

    # 9. Run G1 coverage assertion
    print("\n[9/9] Running G1 coverage assertion...")
    _check_g1(dev_df, cov_df)

    dev_months = dev_df["datetime"].dt.to_period("M").nunique()
    print(f"\n  DEV  span  : {len(dev_df):,} trades | {dev_months} calendar months | "
          f"{dev_df['datetime'].min().date()} -> {dev_df['datetime'].max().date()}")
    print(f"  VAULT span : {len(vault_df):,} trades | "
          f"{vault_df['datetime'].min().date()} -> {vault_df['datetime'].max().date()}")

    # Save final panel
    out_path = os.path.join(OUT_DIR, "trade_panel.parquet")
    panel_df.to_parquet(out_path, index=False)
    print(f"\n  Panel saved -> {out_path}")

    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    sha_path = os.path.join(OUT_DIR, "trade_panel_sha256.txt")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(sha)
    print(f"  SHA-256: {sha}")

    # Save feature list for downstream scripts
    META_COLS = ["model", "datetime", "ticker", "side", "trade_return", "Query_ID", "span", "y", "proposed_by"]
    feature_cols = [c for c in panel_df.columns if c not in META_COLS]
    meta_out = {
        "mv_version":     "mv_v10_v19",
        "panel_sha256":   sha,
        "dev_cutoff":     str(DEV_CUTOFF.date()),
        "n_dev_trades":   int(len(dev_df)),
        "n_dev_months":   int(dev_months),
        "n_vault_trades": int(len(vault_df)),
        "feature_cols":   feature_cols,
        "standard_cols":  STANDARD_COLS,
        "core_cols":      CORE_COLS,
    }
    meta_path = os.path.join(OUT_DIR, "panel_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  Panel metadata -> {meta_path}")

    # Correlation checks
    print("\n  Feature correlation with trade_return:")
    for fc in feature_cols:
        try:
            corr = panel_df[fc].corr(panel_df["trade_return"])
            flag = " ⚠️ HIGH" if abs(corr) >= 0.95 else ""
            print(f"    {fc:<35}: {corr:+.4f}{flag}")
            if abs(corr) >= 0.95:
                raise RuntimeError(f"[LEAKAGE] Feature '{fc}' has correlation {corr:.4f} with trade_return.")
        except TypeError:
            pass

    print("=" * 70)
    print("MV2-R0 COMPLETED — v10-v19 trade panel created successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
