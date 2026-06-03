"""
Training for the 10:30 AM Momentum Strategy (V3).

Layer A: XGBClassifier predicting Nifty UP probability.
Layer B: XGBRegressor predicting vol-normalized residual return.
"""

import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from scripts.strategy_1030.config import (
    DATA_DIR,
    MODEL_DIR,
    LAYER_A_FEATURES,
    LAYER_B_FEATURES,
)


def get_walk_forward_folds(df, date_col="Date", train_months=6, val_months=2, test_months=2):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    df["Month"] = df[date_col].dt.to_period("M").astype(str)
    unique_months = sorted(df["Month"].unique())
    total = len(unique_months)
    window = train_months + val_months + test_months

    print(f"  Dataset covers {total} months: {unique_months[0]} to {unique_months[-1]}")

    folds = []
    step = test_months
    for start in range(0, total - window + 1, step):
        train = unique_months[start : start + train_months]
        val = unique_months[start + train_months : start + train_months + val_months]
        test = unique_months[start + train_months + val_months : start + window]
        folds.append({"train": train, "val": val, "test": test})

    print(f"  Generated {len(folds)} walk-forward folds")
    return folds


def train_layer_a():
    """Trains Layer A (Market Filter) with XGBClassifier."""
    print("\n=== Training Layer A: Market Filter (V3 Classifier) ===")

    market_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_market.csv"))
    market_df = market_df.dropna(subset=LAYER_A_FEATURES + ["Nifty_Up"])
    market_df = market_df.sort_values("Date")
    print(f"  Market dataset: {len(market_df)} rows, {len(LAYER_A_FEATURES)} features")

    folds = get_walk_forward_folds(market_df)

    metadata = {"features": LAYER_A_FEATURES, "folds": []}
    all_test_preds = []

    for fold_i, fold in enumerate(folds):
        market_df["Month"] = pd.to_datetime(market_df["Date"]).dt.to_period("M").astype(str)

        train_data = market_df[market_df["Month"].isin(fold["train"])]
        val_data = market_df[market_df["Month"].isin(fold["val"])]
        test_data = market_df[market_df["Month"].isin(fold["test"])]

        if len(train_data) < 50 or len(test_data) < 5:
            continue

        X_train, y_train = train_data[LAYER_A_FEATURES], train_data["Nifty_Up"]
        X_val, y_val = val_data[LAYER_A_FEATURES], val_data["Nifty_Up"]
        X_test, y_test = test_data[LAYER_A_FEATURES], test_data["Nifty_Up"]

        # Predict binary target (1 for Up, 0 for Down)
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Get probability of UP class
        probs = model.predict_proba(X_test)[:, 1]
        preds_binary = (probs > 0.5).astype(int)
        
        accuracy = np.mean(preds_binary == y_test.values)

        model.save_model(
            os.path.join(MODEL_DIR, "market_filter", f"xgb_market_fold_{fold_i}.json")
        )

        for date, prob, actual in zip(test_data["Date"], probs, y_test.values):
            all_test_preds.append({"Date": date, "Prob_Up": prob, "Actual": actual})

        fold_metrics = {
            "fold": fold_i,
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "accuracy": round(float(accuracy), 4),
        }
        metadata["folds"].append({"fold": fold_i, "test_months": fold["test"], "metrics": fold_metrics})
        print(f"  Fold {fold_i:2d}: Train={len(train_data):4d}, Test={len(test_data):3d}, "
              f"Acc={accuracy:.1%}")

    if all_test_preds:
        oos_df = pd.DataFrame(all_test_preds)
        preds_binary = (oos_df["Prob_Up"] > 0.5).astype(int)
        oos_acc = np.mean(preds_binary == oos_df["Actual"])
        print(f"\n  Aggregate OOS Accuracy={oos_acc:.1%}")
        metadata["aggregate_oos"] = {
            "accuracy": round(float(oos_acc), 4),
            "n_predictions": len(oos_df),
        }

    with open(os.path.join(MODEL_DIR, "market_filter", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n  Training final production model on all data...")
    X_all, y_all = market_df[LAYER_A_FEATURES], market_df["Nifty_Up"]
    final_model = XGBClassifier(
        n_estimators=200, learning_rate=0.02, max_depth=3,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42,
    )
    final_model.fit(X_all, y_all)
    final_model.save_model(os.path.join(MODEL_DIR, "market_filter", "xgb_market.json"))
    print("  Final Layer A model saved.")


def train_layer_b():
    """Trains Layer B (Stock Selector) with unified residual XGBRegressor."""
    print("\n=== Training Layer B: Stock Selector (V3 Residuals) ===")

    stocks_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_stocks.csv"))
    stocks_df = stocks_df.dropna(subset=LAYER_B_FEATURES + ["Target"])
    stocks_df = stocks_df.sort_values(["Date", "Ticker"])
    print(f"  Stock dataset: {len(stocks_df)} rows, {stocks_df['Date'].nunique()} dates, "
          f"{stocks_df['Ticker'].nunique()} tickers, {len(LAYER_B_FEATURES)} features")

    folds = get_walk_forward_folds(stocks_df)
    metadata = {"features": LAYER_B_FEATURES, "folds": []}

    for fold_i, fold in enumerate(folds):
        stocks_df["Month"] = pd.to_datetime(stocks_df["Date"]).dt.to_period("M").astype(str)

        train_data = stocks_df[stocks_df["Month"].isin(fold["train"])].sort_values("Date")
        val_data = stocks_df[stocks_df["Month"].isin(fold["val"])].sort_values("Date")
        test_data = stocks_df[stocks_df["Month"].isin(fold["test"])].sort_values("Date")

        if len(train_data) < 500 or len(test_data) < 100:
            continue

        X_train, y_train = train_data[LAYER_B_FEATURES], train_data["Target"]
        X_val, y_val = val_data[LAYER_B_FEATURES], val_data["Target"]
        X_test, y_test = test_data[LAYER_B_FEATURES], test_data["Target"]

        # Unified model predicts normalized residual returns
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=4,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_test)
        
        test_results = test_data[["Date", "Ticker", "Target"]].copy()
        test_results["Pred_Score"] = preds

        # Rank correlation (expected to be small but positive, e.g. 0.03 to 0.07)
        spearman = test_results.groupby("Date").apply(
            lambda g: g["Pred_Score"].corr(g["Target"], method="spearman")
        ).mean()

        model.save_model(
            os.path.join(MODEL_DIR, "stock_selector", f"xgb_residual_fold_{fold_i}.json")
        )

        fold_metrics = {
            "fold": fold_i,
            "train_dates": len(train_data["Date"].unique()),
            "test_dates": len(test_data["Date"].unique()),
            "spearman_residual": round(float(spearman) if not np.isnan(spearman) else 0.0, 4),
        }
        metadata["folds"].append({"fold": fold_i, "test_months": fold["test"], "metrics": fold_metrics})
        print(f"  Fold {fold_i:2d}: Train={fold_metrics['train_dates']:4d}d, "
              f"Test={fold_metrics['test_dates']:3d}d, "
              f"Spearman rho={spearman:.3f}")

    with open(os.path.join(MODEL_DIR, "stock_selector", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n  Training final production model on all data...")
    X_all, y_all = stocks_df[LAYER_B_FEATURES], stocks_df["Target"]

    model = XGBRegressor(
        n_estimators=200, learning_rate=0.02, max_depth=4,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42,
    )
    model.fit(X_all, y_all)
    model.save_model(os.path.join(MODEL_DIR, "stock_selector", "xgb_residual.json"))

    print("  Final Layer B model saved.")


if __name__ == "__main__":
    train_layer_a()
    train_layer_b()
