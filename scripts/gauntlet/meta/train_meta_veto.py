import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import hashlib
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

def build_single_purged_split(df, train_pct=0.6, embargo_days=3):
    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)
    unique_dates = df["datetime"].dt.date.unique()
    n_dates = len(unique_dates)
    
    if n_dates < 5:
        raise ValueError("Too few unique trading days to construct train/val split.")
        
    split_idx = int(n_dates * train_pct)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    
    val_start = min(val_dates)
    embargo_cutoff = pd.to_datetime(val_start) - pd.Timedelta(days=embargo_days)
    
    train_idx = df[(df["datetime"].dt.date.isin(train_dates)) & (df["datetime"] < embargo_cutoff)].index.tolist()
    val_idx = df[df["datetime"].dt.date.isin(val_dates)].index.tolist()
    
    return train_idx, val_idx

def evaluate_threshold_oof(oof_df, features, target_col, pct_floor=0.25):
    best_theta = 0.50
    best_net_return = -9999.0
    best_keep_pct = 0.0
    
    # Grid search over theta
    thetas = np.arange(0.10, 0.90, 0.01)
    for theta in thetas:
        # Kept trades
        kept_mask = oof_df["oof_prob"] >= theta
        n_kept = kept_mask.sum()
        keep_pct = n_kept / len(oof_df)
        
        if keep_pct < pct_floor:
            continue
            
        # Compute mean trade return of kept trades (in bps, raw - 10bps cost)
        # Note: trade_return in the panel is simple return. Convert to bps.
        # trade_return - 10bps (0.0010)
        net_rets = oof_df.loc[kept_mask, "trade_return"] - 0.0010
        mean_net_bps = net_rets.mean() * 10000.0
        
        if mean_net_bps > best_net_return:
            best_net_return = mean_net_bps
            best_theta = theta
            best_keep_pct = keep_pct
            
    return float(best_theta), float(best_net_return), float(best_keep_pct)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Meta-Veto Stacking Model (M2)")
    parser.add_argument("--model-type", type=str, choices=["logistic", "gbm"], default="logistic", help="Model type to train")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"M2: TRAINING META-VETO MODEL (Type: {args.model_type.upper()})")
    print("=" * 70)
    
    panel_path = os.path.join("data", "gauntlet", "meta", "trade_panel.parquet")
    if not os.path.exists(panel_path):
        print(f"[FATAL] Trade panel parquet not found at {panel_path}. Run M0 first.")
        sys.exit(1)
        
    df = pd.read_parquet(panel_path)
    
    # Calculate panel SHA256 checksum for pre-registration audit
    with open(panel_path, "rb") as f:
        panel_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Dataset SHA-256: {panel_sha256}")
    
    # Restrict to DEV span only
    dev_df = df[df["span"] == "DEV"].copy().reset_index(drop=True)
    print(f"DEV span dataset size: {len(dev_df):,} rows")
    
    if len(dev_df) == 0:
        print("[FATAL] DEV span is empty. Cannot train.")
        sys.exit(1)
        
    # Define features to use
    all_cols = dev_df.columns
    base_cols = ["model", "datetime", "ticker", "side", "trade_return", "own_score", "own_z", "own_pct", "Query_ID", "span", "y"]
    features = [c for c in all_cols if c not in base_cols]
    features = [c for c in features if dev_df[c].std() > 1e-8]
    
    print(f"Features list ({len(features)}):")
    for feat in features:
        print(f" - {feat}")
        
    # Set target
    target_col = "y"
    
    # 1. Run Purged Validation on DEV to generate Out-Of-Fold (OOF) predictions
    print("\nRunning single purged train/validation split within DEV span...")
    try:
        train_idx, val_idx = build_single_purged_split(dev_df, train_pct=0.6, embargo_days=3)
    except Exception as e:
        print(f"[FATAL] Failed to build train/val split: {e}")
        sys.exit(1)
        
    print(f"Split results: Train={len(train_idx):,} | Val={len(val_idx):,}")
    dev_df["oof_prob"] = np.nan
    
    X_train, y_train = dev_df.loc[train_idx, features], dev_df.loc[train_idx, target_col]
    X_val = dev_df.loc[val_idx, features]
    
    # Scale features using train statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if args.model_type == "logistic":
        model = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", random_state=42)
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_val_scaled)[:, 1]
    else:
        # Tiny Depth-2 GBM to prevent noise-memorization
        model = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        
    dev_df.loc[val_idx, "oof_prob"] = probs
    
    # Drop rows that were never in validation fold (usually the initial block_size training dates)
    oof_df = dev_df.dropna(subset=["oof_prob"]).copy()
    print(f"OOF predictions generated: {len(oof_df):,} trades")
    
    # 2. Calibrate threshold theta on OOF predictions
    print("Calibrating veto threshold theta on DEV OOF predictions...")
    best_theta, best_net_bps, best_keep_pct = evaluate_threshold_oof(oof_df, features, target_col, pct_floor=0.25)
    print(f"Optimal Veto Threshold (theta): {best_theta:.2f}")
    print(f"OOF Kept Trade Keep %         : {best_keep_pct:.1%}")
    print(f"OOF Kept Trade Net Return     : {best_net_bps:+.2f} bps/trade (after 10bps cost)")
    
    # 3. Train final model on the full DEV span dataset
    print("\nTraining final model on full DEV span...")
    X_dev, y_dev = dev_df[features], dev_df[target_col]
    
    scaler_final = StandardScaler()
    X_dev_scaled = scaler_final.fit_transform(X_dev)
    
    if args.model_type == "logistic":
        model_final = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", random_state=42)
        model_final.fit(X_dev_scaled, y_dev)
    else:
        model_final = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        model_final.fit(X_dev_scaled, y_dev)
        
    # Compute feature coefficients/importance for audit transparency
    print("\nModel Audit Coefficients:")
    if args.model_type == "logistic":
        coefs = model_final.coef_[0]
        for name, val in zip(features, coefs):
            print(f" - {name:<20}: {val:+.4f}")
    else:
        importances = model_final.feature_importances_
        for name, val in zip(features, importances):
            print(f" - {name:<20}: {val:.4f}")

    # 4. Save model and metadata (models/meta_veto_v1/)
    model_dir = os.path.join("models", "meta_veto_v1")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    meta_path = os.path.join(model_dir, "metadata.json")
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model_final, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler_final, scaler_path)
    
    # Compute model binary hash for audit trail
    with open(model_path, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    metadata = {
        "model_type": args.model_type,
        "features": features,
        "theta": best_theta,
        "dev_oof_keep_pct": best_keep_pct,
        "dev_oof_net_return_bps": best_net_bps,
        "panel_sha256": panel_sha256,
        "model_sha256": model_sha256,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # Checksum verification pre-registration print
    print("\n" + "=" * 50)
    print("FROZEN METADATA FOR M4 PRE-REGISTRATION")
    print(f"Metadata file: {meta_path}")
    print(f"Model hash   : {model_sha256}")
    print(f"Threshold    : {best_theta}")
    print("=" * 50)
    print("M2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
