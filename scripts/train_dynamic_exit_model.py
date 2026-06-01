"""
scripts/train_dynamic_exit_model.py
Trains the Dynamic Exit Engine (XGBoost Regressor) to predict the Forward Peak Return.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("=" * 60)
    print("TRAINING DYNAMIC EXIT ENGINE")
    print("=" * 60)
    
    data_path = "data/dynamic_exit_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} intra-trade state records.")
    
    features = [
        'side', 'bars_held', 'unrealized_pnl', 
        'current_conv', 'current_rank', 
        'conv_delta', 'rank_delta'
    ]
    target = 'forward_peak_return'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=20, verbose_eval=False)
    
    preds = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n--- Model Evaluation ---")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"R^2:  {r2:.5f}")
    
    # Classification check: Did it correctly predict if there was *any* positive forward return?
    # We define a peak as forward_peak_return < 0.001 (less than 0.1% left)
    # If the model predicts < 0.001, it says SELL.
    y_test_class = y_test > 0.001
    preds_class = preds > 0.001
    accuracy = (y_test_class == preds_class).mean()
    print(f"\nPeak Detection Accuracy (Threshold 0.1%): {accuracy*100:.2f}%")
    
    # Save Model
    os.makedirs("models/ide", exist_ok=True)
    model.save_model("models/ide/dynamic_exit_model.json")
    with open("models/ide/dynamic_exit_features.txt", "w") as f:
        f.write(",".join(features))
        
    print("\nModel saved to models/ide/dynamic_exit_model.json")
    
if __name__ == "__main__":
    main()
