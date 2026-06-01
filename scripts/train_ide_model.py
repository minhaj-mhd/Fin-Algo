"""
scripts/train_ide_model.py
Trains the Intelligent Decision Engine (IDE) models using XGBoost.
We train two models:
1. Filter Model: Uses only t0 (entry) features to predict success probability.
2. Manager Model: Uses t0 + t1 (first 15m) features to update success probability.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, classification_report


def main():
    print("=" * 60)
    print("TRAINING INTELLIGENT DECISION ENGINE")
    print("=" * 60)
    
    data_path = "data/ide_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")
    
    # Define features
    t0_features = ['side', 't0_conv', 't0_rank'] + [c for c in df.columns if c.startswith('feat_')]
    t1_features = t0_features + ['t1_conv_delta', 't1_rank_delta', 't1_mfe', 't1_mae']
    
    label = 'label'
    
    # ---------------------------------------------------------
    # 1. Train Filter Model (Pre-Trade)
    # ---------------------------------------------------------
    print("\n--- Training IDE Filter Model (t0) ---")
    X0 = df[t0_features]
    y = df[label]
    
    # With a small dataset, we'll use a simple CV or just evaluate on train for now
    # to see if it learns anything. Since it's only 72 rows, we don't have enough for a rich test set.
    # But let's split anyway.
    X0_train, X0_test, y_train, y_test = train_test_split(X0, y, test_size=0.3, random_state=42, stratify=y)
    
    dtrain0 = xgb.DMatrix(X0_train, label=y_train)
    dtest0 = xgb.DMatrix(X0_test, label=y_test)
    dall0 = xgb.DMatrix(X0, label=y)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    evals = [(dtrain0, 'train'), (dtest0, 'test')]
    filter_model = xgb.train(params, dtrain0, num_boost_round=50, evals=evals, early_stopping_rounds=10, verbose_eval=False)
    
    # Evaluate Filter Model
    preds0 = filter_model.predict(dtest0)
    ap0 = average_precision_score(y_test, preds0)
    print(f"Filter Model Test Average Precision: {ap0:.3f}")
    
    # Save Model
    os.makedirs("models/ide", exist_ok=True)
    filter_model.save_model("models/ide/filter_model.json")
    with open("models/ide/filter_features.txt", "w") as f:
        f.write(",".join(t0_features))
        
    # ---------------------------------------------------------
    # 2. Train Manager Model (In-Trade)
    # ---------------------------------------------------------
    print("\n--- Training IDE Manager Model (t0 + t1) ---")
    X1 = df[t1_features]
    X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=42, stratify=y)
    
    dtrain1 = xgb.DMatrix(X1_train, label=y_train)
    dtest1 = xgb.DMatrix(X1_test, label=y_test)
    dall1 = xgb.DMatrix(X1, label=y)
    
    evals1 = [(dtrain1, 'train'), (dtest1, 'test')]
    manager_model = xgb.train(params, dtrain1, num_boost_round=50, evals=evals1, early_stopping_rounds=10, verbose_eval=False)
    
    # Evaluate Manager Model
    preds1 = manager_model.predict(dtest1)
    ap1 = average_precision_score(y_test, preds1)
    print(f"Manager Model Test Average Precision: {ap1:.3f}")
    
    # Compare predictions
    df_eval = X1_test.copy()
    df_eval['True Label'] = y_test
    df_eval['Filter_Prob'] = preds0
    df_eval['Manager_Prob'] = preds1
    print("\nSample Test Set Predictions:")
    print(df_eval[['True Label', 'Filter_Prob', 'Manager_Prob', 't1_mfe', 't1_mae']].head(10).round(3))
    
    # Save Model
    manager_model.save_model("models/ide/manager_model.json")
    with open("models/ide/manager_features.txt", "w") as f:
        f.write(",".join(t1_features))
        
    print("\nModels saved to models/ide/")

if __name__ == "__main__":
    main()
