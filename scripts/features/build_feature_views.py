"""
Feature View Builder — Phase 2.

Joins TBM labels with the 1h feature dataset, drops time-crutch features,
and splits remaining features into 3 decorrelated views:
  A — Mean-Reversion  (microstructure, oscillators)
  B — Trend/Momentum  (price dynamics, MA distances)
  C — Volatility/Structure (vol, breadth, regime)

Reads:
  data/ranking_data_upstox_1h_v3_3y.csv
  data/tbm_labels_1h.parquet

Writes:
  data/tbm_feature_views/A_meanrev.parquet
  data/tbm_feature_views/B_trend.parquet
  data/tbm_feature_views/C_vol.parquet

Usage:
    python scripts/features/build_feature_views.py
"""

import os, sys
import numpy as np
import pandas as pd

# ── feature definitions ───────────────────────────────────────────────────────

# Time features — DROP (proven overfit clock in v18/v19)
TIME_FEATS = ['Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close']

# Metadata / targets — never features
META_COLS = ['DateTime', 'Ticker', 'Query_ID', 'Next_Hour_Return',
             'Open', 'High', 'Low', 'Close', 'Volume']

# View A: Mean-reversion / microstructure / oscillators
VIEW_A = [
    'IBS', 'IBS_3', 'Buy_Pressure',
    'Upper_Shadow', 'Lower_Shadow',
    'VWAP_Dist', 'Intraday_Return',
    'PercentB', 'Dist_BB_Upper', 'Dist_BB_Lower',
    'Stoch_K', 'Stoch_D',
    'WPR_14', 'RSI_14',
    'CMF_20', 'OBV_Dist',
    'Elder_Bull', 'Elder_Bear',
    'Price_Zscore',
    'Direction_Consistency_3', 'Direction_Consistency_5',
]

# View B: Trend / momentum / price dynamics
VIEW_B = [
    'Return', 'Log_Return', 'OC_Range',
    'ROC_12', 'MOM_12_pct',
    'PPO', 'PPO_Signal', 'PPO_Hist',
    'TRIX_15', 'Dist_DPO_20',
    'Dist_SMA_6', 'Dist_SMA_12', 'Dist_SMA_50',
    'Dist_EMA_12', 'Dist_EMA_24', 'Dist_HMA_12',
    'Dist_Donchian_Upper', 'Dist_Donchian_Lower',
    'Vortex_Plus', 'Vortex_Minus',
    'Up_Streak', 'Down_Streak',
    'Return_lag1', 'Return_lag2', 'Return_lag3',
    'Return_Accel', 'Price_Accel',
    'Alpha_3H', 'Alpha_6H',
    'Market_Mean_Return', 'Relative_Return',
]

# View C: Volatility / structure / regime
VIEW_C = [
    'HL_Range',
    'BB_Width', 'Donchian_Width', 'Keltner_Width',
    'Dist_Keltner_Upper', 'Dist_Keltner_Lower',
    'Rolling_Skew', 'Rolling_Kurt',
    'Volume_Change', 'Volume_Zscore', 'RVOL', 'Dollar_Volume', 'PVO',
    'Ultimate_Osc', 'CCI_20',
    'Dist_52W_High', 'Dist_52W_Low',
    'RSI_lag1', 'RSI_lag2', 'RSI_lag3', 'RSI_Momentum',
    'Volume_Zscore_lag1', 'Volume_Zscore_lag2', 'Volume_Zscore_lag3',
    'OC_Range_lag1', 'OC_Range_lag2', 'OC_Range_lag3',
    'Market_Mean_Volatility', 'Relative_Volatility',
]

FEAT_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
LABELS_FILE = 'data/tbm_labels_1h.parquet'
OUT_DIR     = 'data/tbm_feature_views'


# ── helpers ───────────────────────────────────────────────────────────────────

def impute_train_stats(X: np.ndarray, is_train: np.ndarray) -> np.ndarray:
    """Fill NaNs using TRAIN-FOLD column means only. Returns imputed copy."""
    X_out = X.copy()
    train_means = np.nanmean(X[is_train], axis=0)
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X_out[:, ci])
        if bad.any():
            X_out[bad, ci] = train_means[ci] if np.isfinite(train_means[ci]) else 0.0
    return X_out


def build_view(df_merged: pd.DataFrame, view_cols: list[str], view_name: str) -> pd.DataFrame:
    available = [c for c in view_cols if c in df_merged.columns]
    missing   = [c for c in view_cols if c not in df_merged.columns]
    if missing:
        print(f"  [{view_name}] Missing {len(missing)} features (not in 1h CSV): {missing[:5]}...")
    print(f"  [{view_name}] Using {len(available)}/{len(view_cols)} features")

    key_cols = ['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
                'entry_price', 'atr', 'R', 'weight', 'YearMonth']
    out = df_merged[key_cols + available].copy()
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Feature View Builder")
    print("=" * 64)

    # Load 1h features
    print(f"\nLoading {FEAT_FILE} ...")
    df_feat = pd.read_csv(FEAT_FILE)
    print(f"  {df_feat.shape[0]:,} rows × {df_feat.shape[1]} cols")

    # Normalise Ticker: strip .NS suffix if present
    df_feat['Ticker'] = df_feat['Ticker'].str.replace(r'\.NS$', '', regex=True)

    # Parse DateTime
    df_feat['DateTime'] = pd.to_datetime(df_feat['DateTime'])

    # Add YearMonth for fold splitting
    df_feat['YearMonth'] = df_feat['DateTime'].dt.to_period('M').astype(str)

    # Load labels
    print(f"\nLoading {LABELS_FILE} ...")
    df_labels = pd.read_parquet(LABELS_FILE)
    print(f"  {df_labels.shape[0]:,} label rows")

    df_labels['DateTime'] = pd.to_datetime(df_labels['DateTime'])

    # Merge on DateTime × Ticker
    print("\nMerging labels with features ...")
    df_merged = df_feat.merge(
        df_labels[['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
                   'entry_price', 'atr', 'R', 'weight']],
        on=['DateTime', 'Ticker'],
        how='inner',
    )
    print(f"  Merged: {df_merged.shape[0]:,} rows ({df_merged.shape[0]/len(df_feat)*100:.1f}% of feature rows)")

    label_vc = df_merged['label'].value_counts(normalize=True).sort_index()
    print(f"  Label balance → SL:{label_vc.get(0,0):.1%}  TP:{label_vc.get(1,0):.1%}  TO:{label_vc.get(2,0):.1%}")

    # Verify time features will be dropped
    time_found = [c for c in TIME_FEATS if c in df_merged.columns]
    print(f"\n  Time features to DROP (D11): {time_found}")

    # Build and save each view
    os.makedirs(OUT_DIR, exist_ok=True)

    for view_name, view_cols in [('A_meanrev', VIEW_A), ('B_trend', VIEW_B), ('C_vol', VIEW_C)]:
        df_view = build_view(df_merged, view_cols, view_name)
        out_path = os.path.join(OUT_DIR, f'{view_name}.parquet')
        df_view.to_parquet(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✅ {view_name}: {df_view.shape[0]:,} rows × {df_view.shape[1]} cols → {out_path} ({size_mb:.1f} MB)")

    print(f"\nDone. Views in {OUT_DIR}/")
    print("Next: python scripts/validation/purged_wf_tbm.py")


if __name__ == '__main__':
    main()
