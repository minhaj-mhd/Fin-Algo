import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import confusion_matrix

def main():
    print("Loading data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
    unique_months = sorted(df['YearMonth'].unique())
    split_idx = int(len(unique_months) * 0.8)
    test_months = unique_months[split_idx-1:] 
    
    df_test = df[df['YearMonth'].isin(test_months)].copy()

    with open('models/research/v24_binary_1h_top20/metadata.json') as f:
        v24_meta = json.load(f)
    v24_feats_long = v24_meta.get('features_long', v24_meta['features'])
    v24_feats_short = v24_meta.get('features_short', v24_meta['features'])
    
    v24_long = xgb.Booster()
    v24_long.load_model('models/research/v24_binary_1h_top20/xgb_long_model.json')
    v24_short = xgb.Booster()
    v24_short.load_model('models/research/v24_binary_1h_top20/xgb_short_model.json')

    print("Scoring test set...")
    d24_l = xgb.DMatrix(df_test[v24_feats_long].values)
    d24_s = xgb.DMatrix(df_test[v24_feats_short].values)
    
    long_probs = v24_long.predict(d24_l)
    short_probs = v24_short.predict(d24_s)
    
    y_true_returns = df_test['Next_Hour_Return'].values
    y_true_long = (y_true_returns > 0).astype(int)
    y_true_short = (y_true_returns < 0).astype(int)

    def print_cm(name, y_true, probs, threshold):
        y_pred = (probs > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name} (Threshold > {threshold})")
        print(f"True Positives (Winning trades we took):  {tp}")
        print(f"False Positives (Losing trades we took): {fp}")
        print(f"True Negatives (Losing trades avoided):  {tn}")
        print(f"False Negatives (Winning trades missed): {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Precision (Win Rate): {precision*100:.1f}%")
        print(f"Recall: {recall*100:.1f}%")

    print("\n--- BASE V24 MODEL CONFUSION METRICS (ALL DATA) ---")
    print_cm("Long Model", y_true_long, long_probs, 0.50)
    print_cm("Long Model", y_true_long, long_probs, 0.535)
    
    print_cm("Short Model", y_true_short, short_probs, 0.50)
    print_cm("Short Model", y_true_short, short_probs, 0.55)

if __name__ == '__main__':
    main()
