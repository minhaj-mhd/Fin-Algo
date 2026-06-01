import pandas as pd
import xgboost as xgb
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def eval_model(name, data_path, meta_path, long_model_path, short_model_path, friction=0.0006):
    print(f"\n--- {name} RAW MODEL (1 Bar Hold) ---")
    df = pd.read_csv(data_path)
    df = df[df['DateTime'].str.startswith('2026-05')].copy()
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    features = meta['features']
    df = df.dropna(subset=features).copy()
    dmatrix = xgb.DMatrix(df[features])

    l = xgb.Booster()
    l.load_model(long_model_path)
    df['long_conv'] = l.predict(dmatrix)

    s = xgb.Booster()
    s.load_model(short_model_path)
    df['short_conv'] = s.predict(dmatrix)

    df['long_rank'] = df.groupby('DateTime')['long_conv'].rank(ascending=False)
    df['short_rank'] = df.groupby('DateTime')['short_conv'].rank(ascending=False)

    df = df.sort_values(['Ticker', 'DateTime'])
    df['Close_T1'] = df.groupby('Ticker')['Close'].shift(-1)
    df['Gross_Long_Ret_1B'] = (df['Close_T1'] / df['Close']) - 1.0
    df['Gross_Short_Ret_1B'] = 1.0 - (df['Close_T1'] / df['Close'])
    df['Net_Long_Ret_1B'] = df['Gross_Long_Ret_1B'] - friction
    df['Net_Short_Ret_1B'] = df['Gross_Short_Ret_1B'] - friction
    df = df.dropna(subset=['Close_T1'])

    for r in [1, 3]:
        for side in ['long', 'short']:
            subset = df[df[f'{side}_rank'] <= r]
            col_gross = 'Gross_Long_Ret_1B' if side == 'long' else 'Gross_Short_Ret_1B'
            col_net = 'Net_Long_Ret_1B' if side == 'long' else 'Net_Short_Ret_1B'
            
            rets_gross = subset[col_gross].values
            rets_net = subset[col_net].values
            
            if len(rets_gross) > 0:
                wr_gross = np.sum(rets_gross > 0) / len(rets_gross) * 100
                wr_net = np.sum(rets_net > 0) / len(rets_net) * 100
                tot_gross = np.sum(rets_gross) * 100
                tot_net = np.sum(rets_net) * 100
                
                print(f'{name} Top {r} {side.capitalize()}: Trades={len(rets_gross)}')
                print(f'  Gross (0 slippage)   -> WR={wr_gross:.1f}%, Total Ret={tot_gross:.2f}%')
                print(f'  Net (0.06% slippage) -> WR={wr_net:.1f}%, Total Ret={tot_net:.2f}%\n')

# 15M Model
eval_model("15M", 'data/ranking_data_upstox_15min_1y.csv', 'models/v1_15min/metadata.json', 'models/v1_15min/xgb_long_model.json', 'models/v1_15min/xgb_short_model.json')

# 30M Model
eval_model("30M", 'data/ranking_data_upstox_30min_1y.csv', 'models/v1_30min/metadata.json', 'models/v1_30min/xgb_long_model.json', 'models/v1_30min/xgb_short_model.json')

# 1H Model (Wait, the data is ranking_data_upstox_3y.csv and model is v8_upstox_3y)
eval_model("1H", 'data/ranking_data_upstox_3y.csv', 'models/v8_upstox_3y/metadata.json', 'models/v8_upstox_3y/xgb_long_model.json', 'models/v8_upstox_3y/xgb_short_model.json')
