"""
MV2-R0: Coverage-Rectified Trade Panel Builder
================================================
Root cause of the void: the v1 builder inner-joined ALL sources, collapsing
the DEV span to the ~30-day window where v8, v2_15m, daily_v2, daily_v3,
v10_d4, AND v11 all simultaneously had OOS predictions.

MV2 fixes:
  - Core features (row-drop if absent):  own_score, own_z, own_pct, trade_return
  - Standard features (NaN-tolerant):    daily_v2/v3 scores+sentiment, cross-TF,
                                          VIX/macro — median-imputed from train
                                          folds only + paired _missing indicator
  - Dropped entirely:                    sibling 1h cols (v10_d4, v11) — they
                                          caused the overlap collapse and are
                                          recoverable via cross-TF anyway
  - own_score/z/pct included in meta-features (were wrongly excluded in v1)
  - Family identity: model_is_v8, side_is_long one-hot indicators
  - G1 assertion: DEV >= 12 distinct months AND >= 5,000 trades (code, not prose)
  - Month x coverage matrix printed to stdout and saved as CSV
  - Output: data/gauntlet/meta/mv2/trade_panel.parquet
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
OUT_DIR = os.path.join("data", "gauntlet", "meta", "mv2_clean")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_run_id_in_ledger(run_dir: str) -> None:
    """
    G6 Guardrail: Verifies that the run_id (the folder name of run_dir)
    exists as a 'completed' event in the central gauntlet ledger.
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
                        found = True
                        break
            except Exception:
                pass
                
    if not found:
        raise RuntimeError(
            f"[G6 VIOLATED] Run ID '{run_id}' (from path {run_dir}) was not found as a completed run in the central ledger.\n"
            "This run is invalid or lookahead-contaminated. Panel building aborted."
        )


def load_preds(run_dir: str, dataset_path: str, usecols: list) -> pd.DataFrame:
    """Load preds.npz + matching CSV rows for the test-set indices."""
    verify_run_id_in_ledger(run_dir)
    npz_path = os.path.join(run_dir, "preds.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"preds.npz not found: {npz_path}")
    npz = np.load(npz_path)
    df_raw = pd.read_csv(dataset_path, usecols=usecols)
    test_idx = npz["idx"]
    df = df_raw.iloc[test_idx].copy()
    if "rl" in npz:
        df["pred_long"] = npz["rl"]
    if "rs" in npz:
        df["pred_short"] = npz["rs"]
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    if df["DateTime"].dt.tz is not None:
        df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    return df


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


def extract_topk_trades(df: pd.DataFrame, K: int, side: str, return_col: str) -> pd.DataFrame:
    """Return top-K rows per Query_ID sorted by the model score."""
    score_col = "pred_long" if side == "long" else "pred_short"
    if score_col not in df.columns:
        return pd.DataFrame()
    df_sorted = df.sort_values(["Query_ID", score_col], ascending=[True, False])
    df_topk   = df_sorted.groupby("Query_ID").head(K).copy()
    df_topk["trade_return"] = df_topk[return_col] if side == "long" else -df_topk[return_col]
    return df_topk


def build_coverage_matrix(sources: dict) -> pd.DataFrame:
    """
    Build a month × source coverage table (row count per month).
    `sources` is a dict of {source_name: DataFrame with 'DateTime' column}.
    """
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
    """
    G1 Guardrail: DEV span must have >= 12 distinct calendar months
    AND >= 5,000 trades. Raises RuntimeError with the coverage matrix
    if violated.
    """
    dev_months = dev_df["datetime"].dt.to_period("M").nunique()
    dev_trades = len(dev_df)
    if dev_months < 12 or dev_trades < 5_000:
        cov_str = coverage_df.to_string(index=False)
        raise RuntimeError(
            f"\n[G1 VIOLATED] DEV span has {dev_months} months and "
            f"{dev_trades:,} trades — need >= 12 months AND >= 5,000 trades.\n"
            f"Coverage matrix:\n{cov_str}\n"
            "Inspect the matrix above to identify the sparse sources. "
            "Do NOT lower the threshold — use the coverage booster (R0.6) instead."
        )
    print(f"[G1 OK] DEV span: {dev_months} months, {dev_trades:,} trades.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MV2-R0: BUILDING COVERAGE-RECTIFIED CANDIDATE TRADE PANEL")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Locate run directories
    # ------------------------------------------------------------------
    print("\n[1/9] Locating Gauntlet run directories...")
    try:
        run_v8       = find_latest_completed_run("v8_upstox_3y")
        run_v2       = find_latest_completed_run("v2_15min_3y")
        run_daily_v2 = find_latest_completed_run("daily_macro_v2")
        run_daily_v3 = find_latest_completed_run("daily_macro_v3")
    except Exception as e:
        print(f"[FATAL] Cannot locate required run directories: {e}")
        sys.exit(1)

    print(f"  v8_upstox_3y   : {run_v8}")
    print(f"  v2_15min_3y    : {run_v2}")
    print(f"  daily_macro_v2 : {run_daily_v2}")
    print(f"  daily_macro_v3 : {run_daily_v3}")
    print("  Note: sibling 1H cols (v10_d4, v11) intentionally DROPPED (MV2-R0 spec).")

    # ------------------------------------------------------------------
    # 2. Load intraday predictions
    # ------------------------------------------------------------------
    print("\n[2/9] Loading intraday model predictions...")
    COLS_1H  = ["DateTime", "Ticker", "Query_ID", "Next_Hour_Return",  "Hour", "DayOfWeek"]
    COLS_15M = ["DateTime", "Ticker", "Query_ID", "Next_15Min_Return", "Hour", "DayOfWeek"]

    df_v8 = load_preds(run_v8, "data/ranking_data_upstox_1h_v3_3y.csv",          COLS_1H)
    df_v2 = load_preds(run_v2, "data/ranking_data_upstox_15min_3y_clean.csv",     COLS_15M)

    print("  [R0.6] G6/MV2-CLEAN enforcement: backfill preds are permanently bypassed to avoid contamination.")

    # Query stats for own-score features (own_z, own_pct)
    df_v8 = add_query_stats(df_v8, "pred_long",  "v8_long")
    df_v8 = add_query_stats(df_v8, "pred_short", "v8_short")
    df_v2 = add_query_stats(df_v2, "pred_long",  "v2_long")
    df_v2 = add_query_stats(df_v2, "pred_short", "v2_short")

    print(f"  v8  1H rows : {len(df_v8):,}  | date range: "
          f"{df_v8['DateTime'].min().date()} -> {df_v8['DateTime'].max().date()}")
    print(f"  v2 15M rows : {len(df_v2):,}  | date range: "
          f"{df_v2['DateTime'].min().date()} -> {df_v2['DateTime'].max().date()}")

    # ------------------------------------------------------------------
    # 3. Load daily macro context predictions
    # ------------------------------------------------------------------
    print("\n[3/9] Loading daily macro context predictions...")
    COLS_DV2 = ["DateTime", "Ticker", "Query_ID",
                "Nifty50_Dist_SMA_200", "VIX_Level", "VIX_Percentile_1Y"]

    df_dv2 = load_preds(run_daily_v2, "data/ranking_data_daily_macro_v2.csv", COLS_DV2)
    df_dv3 = load_preds(run_daily_v3, "data/ranking_data_daily_macro_v3.csv",
                        ["DateTime", "Ticker", "Query_ID"])

    df_dv2 = add_query_stats(df_dv2, "pred_long",  "daily_v2_long")
    df_dv2 = add_query_stats(df_dv2, "pred_short", "daily_v2_short")
    df_dv2 = add_day_sentiment(df_dv2, "pred_long",  "daily_v2_long")
    df_dv2 = add_day_sentiment(df_dv2, "pred_short", "daily_v2_short")

    df_dv3 = add_query_stats(df_dv3, "pred_long",  "daily_v3_long")
    df_dv3 = add_query_stats(df_dv3, "pred_short", "daily_v3_short")
    df_dv3 = add_day_sentiment(df_dv3, "pred_long",  "daily_v3_long")
    df_dv3 = add_day_sentiment(df_dv3, "pred_short", "daily_v3_short")

    print(f"  daily_v2 rows: {len(df_dv2):,}  | dates: "
          f"{df_dv2['DateTime'].min().date()} -> {df_dv2['DateTime'].max().date()}")
    print(f"  daily_v3 rows: {len(df_dv3):,}  | dates: "
          f"{df_dv3['DateTime'].min().date()} -> {df_dv3['DateTime'].max().date()}")

    # ------------------------------------------------------------------
    # 4. Coverage matrix (before any join — shows raw availability per month)
    # ------------------------------------------------------------------
    print("\n[4/9] Building month × source coverage matrix...")
    coverage_sources = {
        "v8_1H":       df_v8.rename(columns={"DateTime": "DateTime"})[["DateTime"]],
        "v2_15M":      df_v2[["DateTime"]],
        "daily_v2":    df_dv2[["DateTime"]],
        "daily_v3":    df_dv3[["DateTime"]],
    }
    cov_df = build_coverage_matrix(coverage_sources)
    cov_path = os.path.join(OUT_DIR, "coverage_matrix.csv")
    cov_df.to_csv(cov_path, index=False)

    print("\n  MONTH × SOURCE COVERAGE (row counts)")
    print("  " + cov_df.to_string(index=False).replace("\n", "\n  "))
    print(f"\n  Coverage matrix saved -> {cov_path}")

    # ------------------------------------------------------------------
    # 5. Extract candidate trades (Top-3 per query per model+side)
    # ------------------------------------------------------------------
    print("\n[5/9] Extracting Top-3 candidate trades per query...")
    trade_rows = []

    def _add_trades(df_src, model_name, side, return_col, z_col, pct_col):
        score_col = "pred_long" if side == "long" else "pred_short"
        topk = extract_topk_trades(df_src, K=3, side=side, return_col=return_col)
        for _, row in topk.iterrows():
            trade_rows.append({
                "model":        model_name,
                "datetime":     row["DateTime"],
                "ticker":       row["Ticker"],
                "side":         side,
                "trade_return": row["trade_return"],
                "own_score":    row[score_col],
                "own_z":        row[z_col],
                "own_pct":      row[pct_col],
                "Query_ID":     row["Query_ID"],
                "hour":         row["Hour"],
                "day_of_week":  row["DayOfWeek"],
            })

    _add_trades(df_v8, "v8_upstox_3y", "long",  "Next_Hour_Return",  "v8_long_z",  "v8_long_pct")
    _add_trades(df_v8, "v8_upstox_3y", "short", "Next_Hour_Return",  "v8_short_z", "v8_short_pct")
    _add_trades(df_v2, "v2_15min_3y",  "long",  "Next_15Min_Return", "v2_long_z",  "v2_long_pct")
    _add_trades(df_v2, "v2_15min_3y",  "short", "Next_15Min_Return", "v2_short_z", "v2_short_pct")

    panel_df = pd.DataFrame(trade_rows)
    print(f"  Extracted {len(panel_df):,} raw candidate trades.")

    # ------------------------------------------------------------------
    # 6. Family identity indicators
    # ------------------------------------------------------------------
    panel_df["model_is_v8"]  = (panel_df["model"] == "v8_upstox_3y").astype(int)
    panel_df["side_is_long"] = (panel_df["side"] == "long").astype(int)

    # ------------------------------------------------------------------
    # 7. Cross-timeframe score (NaN-tolerant — standard tier)
    # ------------------------------------------------------------------
    print("\n[6/9] Joining cross-timeframe scores (NaN-tolerant)...")

    unique_15m_times = sorted(panel_df.loc[panel_df["model"] == "v2_15min_3y", "datetime"].unique())
    unique_1h_times  = sorted(df_v8["DateTime"].unique())

    # 15M trades -> look up latest completed 1H candle (t_1H <= t_15M - 1h)
    t15m_to_t1h = {}
    for t15m in unique_15m_times:
        cutoff = t15m - pd.Timedelta(hours=1)
        matched = [t for t in unique_1h_times if t <= cutoff]
        if matched:
            t15m_to_t1h[t15m] = max(matched)

    panel_df["_mapped_1h_dt"] = pd.NaT
    mask_15m = panel_df["model"] == "v2_15min_3y"
    panel_df.loc[mask_15m, "_mapped_1h_dt"] = panel_df.loc[mask_15m, "datetime"].map(t15m_to_t1h)

    # 1H trades -> look up the 15M candle 15m prior
    panel_df["_mapped_15m_dt"] = pd.NaT
    mask_1h = panel_df["model"] == "v8_upstox_3y"
    panel_df.loc[mask_1h, "_mapped_15m_dt"] = (
        panel_df.loc[mask_1h, "datetime"] - pd.Timedelta(minutes=15)
    )

    # Merge 1H scores -> 15M trades
    df_v8_sub = df_v8[["DateTime", "Ticker", "v8_long_pct", "v8_short_pct"]].copy()
    panel_df = pd.merge(
        panel_df, df_v8_sub,
        left_on=["_mapped_1h_dt", "ticker"],
        right_on=["DateTime", "Ticker"],
        how="left", suffixes=("", "_cf1h")
    ).drop(columns=["DateTime", "Ticker"], errors="ignore")
    panel_df["_ctf_via_1h"] = np.where(
        panel_df["side"] == "long", panel_df["v8_long_pct"], panel_df["v8_short_pct"]
    )
    panel_df = panel_df.drop(columns=["v8_long_pct", "v8_short_pct"], errors="ignore")

    # Merge 15M scores -> 1H trades
    df_v2_sub = df_v2[["DateTime", "Ticker", "v2_long_pct", "v2_short_pct"]].copy()
    panel_df = pd.merge(
        panel_df, df_v2_sub,
        left_on=["_mapped_15m_dt", "ticker"],
        right_on=["DateTime", "Ticker"],
        how="left", suffixes=("", "_cf15m")
    ).drop(columns=["DateTime", "Ticker"], errors="ignore")
    panel_df["_ctf_via_15m"] = np.where(
        panel_df["side"] == "long", panel_df["v2_long_pct"], panel_df["v2_short_pct"]
    )
    panel_df = panel_df.drop(columns=["v2_long_pct", "v2_short_pct"], errors="ignore")

    panel_df["cross_tf_pct"] = np.where(
        panel_df["model"] == "v2_15min_3y",
        panel_df["_ctf_via_1h"],
        panel_df["_ctf_via_15m"]
    )
    panel_df = panel_df.drop(
        columns=["_mapped_1h_dt", "_mapped_15m_dt", "_ctf_via_1h", "_ctf_via_15m"],
        errors="ignore"
    )

    # ------------------------------------------------------------------
    # 8. Daily macro context join (T-1 point-in-time, NaN-tolerant)
    # ------------------------------------------------------------------
    print("[7/9] Joining daily macro context (T-1 PIT, NaN-tolerant)...")

    panel_df["_trade_date"] = panel_df["datetime"].dt.normalize()
    unique_trade_dates = sorted(panel_df["_trade_date"].unique())
    unique_daily_dates = sorted(df_dv2["DateTime"].dt.normalize().unique())

    trade_date_to_daily = {}
    for td in unique_trade_dates:
        prev = [d for d in unique_daily_dates if d < td]
        if prev:
            trade_date_to_daily[td] = max(prev)

    panel_df["_mapped_daily"] = panel_df["_trade_date"].map(trade_date_to_daily)

    # Daily v2
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

    # Daily v3
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

    # ------------------------------------------------------------------
    # 9. Tiered feature policy: core drop + standard _missing indicators
    # ------------------------------------------------------------------
    print("\n[8/9] Applying tiered feature policy...")

    # CORE: drop rows missing any core feature
    CORE_COLS = ["own_score", "own_z", "own_pct", "trade_return"]
    before = len(panel_df)
    panel_df = panel_df.dropna(subset=CORE_COLS)
    after = len(panel_df)
    if before > after:
        print(f"  Dropped {before - after:,} rows missing core features.")

    # STANDARD (NaN-tolerant): add _missing indicator, will be imputed
    # per fold in dev_run.py. Here we just log missingness rates.
    STANDARD_COLS = [
        "cross_tf_pct",
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

    # ------------------------------------------------------------------
    # 10. Target, span, final feature list
    # ------------------------------------------------------------------
    panel_df["y"] = ((panel_df["trade_return"] - COST_RATE) > 0).astype(int)
    panel_df["span"] = np.where(panel_df["datetime"] < DEV_CUTOFF, "DEV", "VAULT")

    dev_df   = panel_df[panel_df["span"] == "DEV"]
    vault_df = panel_df[panel_df["span"] == "VAULT"]

    # G1 check (code assertion, not prose)
    print("\n[9/9] Running G1 coverage assertion...")
    _check_g1(dev_df, cov_df)

    # Summary
    dev_months = dev_df["datetime"].dt.to_period("M").nunique()
    print(f"\n  DEV  span  : {len(dev_df):,} trades | {dev_months} calendar months | "
          f"{dev_df['datetime'].min().date()} -> {dev_df['datetime'].max().date()}")
    print(f"  VAULT span : {len(vault_df):,} trades | "
          f"{vault_df['datetime'].min().date()} -> {vault_df['datetime'].max().date()}")

    # Row counts by model+side
    print("\n  Trades by model and side:")
    for key, cnt in panel_df.groupby(["model", "side"]).size().items():
        print(f"    {key[0]} / {key[1]:<6}: {cnt:,}")

    # Final feature columns (everything except identity/metadata cols)
    META_COLS = ["model", "datetime", "ticker", "side", "trade_return",
                 "Query_ID", "span", "y"]
    feature_cols = [c for c in panel_df.columns if c not in META_COLS]
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for fc in feature_cols:
        print(f"    {fc}")

    # Leakage guard: no feature should correlate > 0.95 with trade_return
    print("\n  Leakage guard (feature vs trade_return correlation):")
    for fc in feature_cols:
        try:
            corr = panel_df[fc].corr(panel_df["trade_return"])
            flag = " ⚠️ HIGH" if abs(corr) >= 0.95 else ""
            print(f"    {fc:<28}: {corr:+.4f}{flag}")
            if abs(corr) >= 0.95:
                raise RuntimeError(
                    f"[LEAKAGE] Feature '{fc}' has correlation {corr:.4f} with "
                    "trade_return — suspected leakage. Abort."
                )
        except TypeError:
            pass  # non-numeric column

    # ------------------------------------------------------------------
    # 11. Save panel
    # ------------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, "trade_panel.parquet")
    panel_df.to_parquet(out_path, index=False)
    print(f"\n  Panel saved -> {out_path}")

    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    sha_path = os.path.join(OUT_DIR, "trade_panel_sha256.txt")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(sha)
    print(f"  SHA-256: {sha}")
    print(f"  Checksum saved -> {sha_path}")

    # Save feature list for downstream scripts
    meta_out = {
        "mv_version":     "mv2",
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

    print("=" * 70)
    print("MV2-R0 COMPLETED — coverage-rectified panel ready for M1 + R1")
    print("=" * 70)


if __name__ == "__main__":
    main()

