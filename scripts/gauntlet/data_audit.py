import os
import hashlib
import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from .contracts import DatasetSpec
from .paths import gauntlet_root

def calculate_sha256(file_path: str) -> str:
    """
    Computes SHA-256 hash of a file in chunks to save memory.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 128):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_parquet_cache_path(sha256_hash: str) -> str:
    """
    Returns the cache path for the Parquet format of the dataset.
    """
    cache_dir = os.path.join(gauntlet_root(), "_cache")
    return os.path.join(cache_dir, f"{sha256_hash}.parquet")

def load_dataset_with_cache(spec: DatasetSpec, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Loads dataset, using the Parquet cache if it exists.
    If cache doesn't exist, loads CSV, saves to Parquet cache, and returns.
    Returns (df, sha256_hash).
    """
    print(f"Calculating SHA-256 for dataset: {spec.path}...")
    sha256_hash = calculate_sha256(spec.path)
    print(f"Dataset hash: {sha256_hash}")
    
    parquet_path = get_parquet_cache_path(sha256_hash)
    if os.path.exists(parquet_path):
        print(f"Loading cached dataset from: {parquet_path}")
        if columns is not None:
            import pyarrow.parquet as pq
            schema = pq.read_schema(parquet_path)
            existing_cols = schema.names
            cols_to_load = [c for c in columns if c in existing_cols]
            df = pd.read_parquet(parquet_path, columns=cols_to_load)
        else:
            df = pd.read_parquet(parquet_path)
    else:
        print(f"Cache miss. Loading CSV dataset from: {spec.path}")
        df = pd.read_csv(spec.path)
        print(f"Caching CSV dataset to Parquet: {parquet_path}")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        if columns is not None:
            existing_cols = df.columns
            cols_to_load = [c for c in columns if c in existing_cols]
            df = df[cols_to_load].copy()
        
    return df, sha256_hash

def audit_dataset(df: pd.DataFrame, spec: DatasetSpec, features: List[str]) -> dict:
    """
    Runs the Stage 0 data audit checks (A0.1 to A0.6).
    Aborts hard (raises AssertionError) on any audit failure.
    Returns a dictionary of audit statistics (coverage stats).
    """
    print("=" * 60)
    print("STAGE 0: RUNNING DATA AUDIT...")
    print("=" * 60)
    
    # A0.1 Schema: all spec.features + label + qid + datetime + ticker present; no duplicate (ticker, datetime) rows
    print("Running A0.1: Schema Check...")
    required_cols = [spec.label_col, spec.qid_col, spec.ticker_col, spec.datetime_col]
    if spec.raw_close_col:
        required_cols.append(spec.raw_close_col)
        
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    for feat in features:
        assert feat in df.columns, f"Missing feature: {feat}"
        
    # Check duplicate (ticker, datetime) rows
    duplicates = df.duplicated(subset=[spec.ticker_col, spec.datetime_col]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (Ticker, DateTime) rows"
    
    # A0.2 Timestamps strictly increasing per ticker; Query_ID <-> unique timestamp bijection; >=5 tickers per query
    print("Running A0.2: Timestamp and Query ID Check...")
    
    # Ensure datetime col is datetime objects
    df_dt = pd.to_datetime(df[spec.datetime_col])
    
    # Timestamps strictly increasing per ticker
    for ticker, group in df.groupby(spec.ticker_col):
        group_dt = pd.to_datetime(group[spec.datetime_col])
        is_increasing = group_dt.is_monotonic_increasing
        assert is_increasing, f"Timestamps not strictly increasing for ticker {ticker}"
        assert len(group_dt) == len(group_dt.unique()), f"Duplicate timestamps for ticker {ticker}"
        
    # Query_ID <-> unique timestamp bijection
    qid_dt_unique = df.groupby(spec.qid_col)[spec.datetime_col].nunique()
    assert (qid_dt_unique == 1).all(), "Query_ID maps to multiple timestamps"
    
    dt_qid_unique = df.groupby(spec.datetime_col)[spec.qid_col].nunique()
    assert (dt_qid_unique == 1).all(), "Timestamp maps to multiple Query_IDs"
    
    # >=5 tickers per query
    query_sizes = df.groupby(spec.qid_col).size()
    min_tickers = query_sizes.min()
    assert min_tickers >= 5, f"Query group with fewer than 5 tickers found (min is {min_tickers})"
    
    # A0.3 Bar-label-side verification: infer convention from last-bar-of-day timestamps and assert it equals spec.bar_label_side
    print("Running A0.3: Bar Label Side Check...")
    df_temp = pd.DataFrame({
        'date': df_dt.dt.date,
        'time': df_dt.dt.time
    })
    max_times_per_day = df_temp.groupby('date')['time'].max()
    close_time = datetime.datetime.strptime(spec.session_close, "%H:%M").time()
    
    inferred_sides = np.where(max_times_per_day >= close_time, "right", "left")
    
    values, counts = np.unique(inferred_sides, return_counts=True)
    most_common_side = values[np.argmax(counts)]
    pct_agreement = np.max(counts) / len(max_times_per_day)
    
    assert pct_agreement >= 0.99, (
        f"Bar-side time convention ambiguity: only {pct_agreement:.2%} of days agree on the bar side. "
        f"Offending dates and times: {max_times_per_day[inferred_sides != most_common_side].to_dict()}"
    )
    
    assert most_common_side == spec.bar_label_side, (
        f"Bar label side mismatch: inferred '{most_common_side}', "
        f"specified '{spec.bar_label_side}' (agreement: {pct_agreement:.2%})"
    )
    
    # A0.4 Label recomputation: rebuild to vectorized row classification
    print("Running A0.4: Vectorized Label Integrity Check...")
    df_sorted = df.copy()
    df_sorted[spec.datetime_col] = pd.to_datetime(df_sorted[spec.datetime_col])
    df_sorted = df_sorted.sort_values([spec.ticker_col, spec.datetime_col]).reset_index(drop=True)
    
    close_col = spec.raw_close_col or "Close"
    assert close_col in df_sorted.columns, f"Close price column '{close_col}' not found for label audit"
    
    df_sorted['target_dt'] = df_sorted[spec.datetime_col] + pd.to_timedelta(spec.label_horizon_bars * spec.bar_minutes, unit='m')
    df_sorted['next_row_dt'] = df_sorted.groupby(spec.ticker_col)[spec.datetime_col].shift(-1)
    
    # Merge to get target Close price
    df_close = df_sorted[[spec.ticker_col, spec.datetime_col, close_col]].rename(
        columns={spec.datetime_col: 'target_dt', close_col: 'target_close'}
    )
    df_sorted = pd.merge(df_sorted, df_close, on=[spec.ticker_col, 'target_dt'], how='left')
    
    is_intra = df_sorted['target_close'].notna()
    same_date = (df_sorted[spec.datetime_col].dt.date == df_sorted['next_row_dt'].dt.date).fillna(False)
    
    df_sorted['row_type'] = 'BOUNDARY'
    df_sorted.loc[is_intra, 'row_type'] = 'INTRA'
    df_sorted.loc[~is_intra & same_date, 'row_type'] = 'UNVERIFIABLE'
    
    # Calculate coverage stats
    total_rows = len(df_sorted)
    count_intra = (df_sorted['row_type'] == 'INTRA').sum()
    count_unverifiable = (df_sorted['row_type'] == 'UNVERIFIABLE').sum()
    count_boundary = (df_sorted['row_type'] == 'BOUNDARY').sum()
    
    pct_verified = count_intra / total_rows
    pct_unverifiable = count_unverifiable / total_rows
    pct_boundary = count_boundary / total_rows
    
    print(f"Label coverage: verified={pct_verified:.2%}, unverifiable={pct_unverifiable:.2%}, boundary={pct_boundary:.2%}")
    
    # Verify INTRA labels (tighten to atol=1e-9 per R7.6)
    df_intra = df_sorted[df_sorted['row_type'] == 'INTRA']
    if len(df_intra) > 0:
        recomputed_labels = df_intra['target_close'] / df_intra[close_col] - 1.0
        label_matches = np.isclose(recomputed_labels, df_intra[spec.label_col], atol=1e-9)
        if not label_matches.all():
            mismatch_idx = np.where(~label_matches)[0][0]
            mismatch_row = df_intra.iloc[mismatch_idx]
            raise AssertionError(
                f"Label mismatch for ticker {mismatch_row[spec.ticker_col]} at {mismatch_row[spec.datetime_col]}: "
                f"recomputed {recomputed_labels.iloc[mismatch_idx]:.9f}, actual {mismatch_row[spec.label_col]:.9f}"
            )
            
    # Anti-overnight check (R2 item 3)
    df_sorted['date'] = df_sorted[spec.datetime_col].dt.date
    df_first = df_sorted.groupby([spec.ticker_col, 'date'])[close_col].first().reset_index()
    df_first['next_date_first_close'] = df_first.groupby(spec.ticker_col)[close_col].shift(-1)
    df_sorted = pd.merge(df_sorted, df_first[[spec.ticker_col, 'date', 'next_date_first_close']], on=[spec.ticker_col, 'date'], how='left')
    
    df_boundary_chk = df_sorted[(df_sorted['row_type'] == 'BOUNDARY') & df_sorted['next_date_first_close'].notna()]
    if spec.bar_minutes >= 1440:
        print("Anti-overnight statistical check: skipped for daily/macro datasets.")
    elif len(df_boundary_chk) > 0:
        overnight_returns = df_boundary_chk['next_date_first_close'] / df_boundary_chk[close_col] - 1.0
        matches = np.abs(df_boundary_chk[spec.label_col] - overnight_returns) < 1e-9
        match_rate = matches.mean()
        print(f"Anti-overnight statistical check: boundary match rate = {match_rate:.2%}")
        assert match_rate < 0.01, f"Overnight label leakage detected! Match rate {match_rate:.2%} exceeds 1% limit."
        
    # Raw source verification for UNVERIFIABLE rows
    df_unverifiable = df_sorted[df_sorted['row_type'] == 'UNVERIFIABLE']
    if len(df_unverifiable) > 0:
        if spec.raw_source_glob:
            print(f"Performing raw-source verification on sample of UNVERIFIABLE rows...")
            import glob
            raw_files = glob.glob(spec.raw_source_glob)
            assert len(raw_files) > 0, f"No raw source parquet files found matching glob '{spec.raw_source_glob}'"
            
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(df_unverifiable.index, size=min(500, len(df_unverifiable)), replace=False)
            df_sample = df_unverifiable.loc[sample_idx].copy()
            df_sample['target_date_str'] = df_sample['target_dt'].dt.strftime('%Y-%m-%d')
            
            verified_count = 0
            for date_str, group in df_sample.groupby('target_date_str'):
                raw_dir = os.path.dirname(spec.raw_source_glob)
                parquet_file = os.path.join(raw_dir, f"{date_str}.parquet")
                if not os.path.exists(parquet_file):
                    raise AssertionError(f"Raw source parquet file not found for date {date_str} at {parquet_file}")
                    
                raw_df = pd.read_parquet(parquet_file)
                raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'])
                
                raw_close_map = dict(zip(zip(raw_df['Ticker'], raw_df['DateTime']), raw_df['Close']))
                
                for idx, row in group.iterrows():
                    ticker_val = row[spec.ticker_col]
                    target_dt_val = row['target_dt']
                    
                    key = (ticker_val, target_dt_val)
                    if key not in raw_close_map:
                        raise AssertionError(f"Target bar {target_dt_val} not found in raw source for ticker {ticker_val}")
                        
                    raw_close = raw_close_map[key]
                    recomputed_label = raw_close / row[close_col] - 1.0
                    assert np.isclose(recomputed_label, row[spec.label_col], atol=1e-9), (
                        f"Raw source label mismatch for ticker {ticker_val} at {row[spec.datetime_col]}: "
                        f"recomputed {recomputed_label:.9f}, actual {row[spec.label_col]:.9f} (raw Close is {raw_close})"
                    )
                    verified_count += 1
            print(f"Successfully verified {verified_count} UNVERIFIABLE rows against raw parquet sources.")
        else:
            assert spec.unverified_label_waiver_reason is not None and len(spec.unverified_label_waiver_reason) > 0, (
                f"Dataset contains {len(df_unverifiable)} ({pct_unverifiable:.2%}) unverifiable rows, "
                f"but no raw_source_glob or unverified_label_waiver_reason was provided."
            )
            print(f"Unverified label waiver used. Reason: {spec.unverified_label_waiver_reason}")
            
    # A0.5 Overnight guard (replaces old time-arithmetic check per R2 item 6)
    if not spec.label_may_cross_session:
        print("Running A0.5: Overnight Guard Check...")
        if len(df_intra) > 0:
            same_date_intra = df_intra[spec.datetime_col].dt.date == df_intra['target_dt'].dt.date
            assert same_date_intra.all(), "INTRA label crosses overnight boundary (different date)"
        terminal_dt = df_dt + pd.to_timedelta(spec.label_horizon_bars * spec.bar_minutes, unit="m")
        assert (terminal_dt.dt.time <= close_time).all(), "Label crosses session close time"
        
    # A0.6 NaN/Inf census: assert label has zero NaNs; no feature is >50% NaN
    print("Running A0.6: NaN/Inf Census...")
    label_nans = df[spec.label_col].isna().sum()
    assert label_nans == 0, f"Label column '{spec.label_col}' contains {label_nans} NaN values"
    
    for feat in features:
        nan_pct = df[feat].isna().mean()
        assert nan_pct <= 0.50, f"Feature '{feat}' contains {nan_pct * 100:.1f}% NaNs (exceeds 50% limit)"
        
    # Hard gate (R2 item 5)
    if pct_verified < 0.80 and not spec.raw_source_glob and not spec.unverified_label_waiver_reason:
        raise AssertionError(f"Dataset verification rate {pct_verified:.2%} is below 80% threshold and no raw-source verification or waiver was provided.")
        
    print("Stage 0: Data Audit PASSED successfully!")
    
    return {
        "pct_verified": float(pct_verified),
        "pct_unverifiable": float(pct_unverifiable),
        "pct_boundary": float(pct_boundary),
        "unverified_label_waiver_reason": spec.unverified_label_waiver_reason
    }
