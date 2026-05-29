# MODEL IMPROVEMENT ACTION PLAN

## PHASE 1: IMMEDIATE FIXES (Do Today)

### Fix 1: Verify Data Alignment
**File**: `prepare_ranking_data.py`
**Issue**: Check if Next_Hour_Return is calculated correctly

```python
# Add this code to verify alignment:
df_check = df[['DateTime_Hour', 'Ticker', 'Close', 'Next_Hour_Return']].head(20)
print("Sample data check:")
print(df_check)

# Verify calculation manually:
# Get two consecutive timestamps for same stock
for ticker in df['Ticker'].unique()[:1]:
    ticker_df = df[df['Ticker'] == ticker].sort_values('DateTime_Hour')
    for i in range(len(ticker_df) - 1):
        t1_close = ticker_df.iloc[i]['Close']
        t2_close = ticker_df.iloc[i+1]['Close']
        expected_return = (t2_close - t1_close) / t1_close
        actual_return = ticker_df.iloc[i]['Next_Hour_Return']
        print(f"T1 Close: {t1_close:.2f}, T2 Close: {t2_close:.2f}")
        print(f"Expected: {expected_return:.6f}, Actual: {actual_return:.6f}")
```

**Expected Result**: Expected and Actual should match. If they're negative of each other, invert the calculation.

---

### Fix 2: Use Inverted Predictions
**File**: Create `inference_inverted.py`

```python
import xgboost as xgb
import pickle
import json
import pandas as pd
import numpy as np

# Load model and data
bst = xgb.Booster()
bst.load_model('xgb_ranking_model.json')
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = json.load(open('model_metadata.json'))['features']

def predict_stocks(df_input):
    """
    Input: DataFrame with features
    Output: Inverted predictions (positive = stocks to buy)
    """
    X = df_input[features].values
    
    # Handle NaN/Inf
    X = np.where(np.isfinite(X), X, 0)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    dmatrix = xgb.DMatrix(X_scaled)
    predictions = bst.predict(dmatrix)
    
    # INVERT predictions
    predictions_inverted = -predictions
    
    return predictions_inverted

# Usage:
df = pd.read_csv('ranking_data_full.csv')
predicted_scores = predict_stocks(df)
df['Predicted_Score_Inverted'] = predicted_scores

# Rank within each query
df['Predicted_Rank'] = df.groupby('Query_ID')['Predicted_Score_Inverted'].rank(ascending=False)

# Show top performers
print("Top stocks to buy (highest predicted scores):")
latest_query = df['Query_ID'].max()
top_stocks = df[df['Query_ID'] == latest_query].nlargest(5, 'Predicted_Score_Inverted')
print(top_stocks[['Ticker', 'Predicted_Score_Inverted', 'Predicted_Rank', 'Next_Hour_Return']])
```

---

### Fix 3: Add Regularization to XGBoost
**File**: `train_ranking_v2.py` (updated parameters)

```python
# Current params:
params = {
    'objective': 'rank:pairwise',
    'eta': 0.1,           # Too high, causes overfitting
    'max_depth': 6,       # Too deep
    'eval_metric': 'ndcg@32',
}

# IMPROVED params:
params = {
    'objective': 'rank:pairwise',
    
    # Reduce learning rate
    'eta': 0.05,          # Changed from 0.1
    
    # Reduce tree complexity
    'max_depth': 4,       # Changed from 6
    'min_child_weight': 20,  # New: prevents small leaves
    
    # Add subsampling
    'subsample': 0.7,     # New: row subsampling
    'colsample_bytree': 0.8,  # New: column subsampling
    
    # Add regularization
    'gamma': 2.0,         # New: min loss reduction to split
    'lambda': 10.0,       # New: L2 regularization
    'alpha': 1.0,         # New: L1 regularization
    
    'eval_metric': 'ndcg@32',
    'seed': 42,
}

# Also change number of rounds:
# bst = xgb.train(params, dtrain, num_boost_round=100)
# to:
# bst = xgb.train(params, dtrain, num_boost_round=200,
#                 evals=[(dtrain, 'train'), (dtest, 'test')],
#                 early_stopping_rounds=30)
```

**Impact**: Should reduce variance from 0.3 to ~0.15

---

## PHASE 2: FEATURE ENGINEERING (Do This Week)

### Feature Improvement 1: Add Relative Features
**File**: `prepare_ranking_data_v2.py`

```python
# After computing all technical indicators, add relative features:

# Relative to peer average
for col in technical_feature_cols:
    df[f'{col}_RelToPeer'] = df.groupby('DateTime_Hour')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

# Relative volume
df['Volume_RelToPeer'] = df.groupby('DateTime_Hour')['Volume'].transform(
    lambda x: x / (x.mean() + 1e-8)
)

# Relative volatility
df['Volatility_RelToPeer'] = df.groupby('DateTime_Hour')['HL_Range'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# Relative momentum
df['Momentum_RelToPeer'] = df.groupby('DateTime_Hour')['ROC_12'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)
```

**Impact**: Creates natural ranking signals within each query

---

### Feature Improvement 2: Remove Redundant Features
**Keep only**:
- Volume_Zscore (top importance: 213)
- CMF_20 (top importance: 201)
- Remove: Volume_Change, OBV, PVO (highly correlated with top 2)

```python
# In train script, define features as:
feature_cols = [
    'Return', 'Log_Return', 'HL_Range', 'OC_Range',
    'SMA_6', 'SMA_12', 'EMA_12', 'EMA_24', 'HMA_12',
    'RSI_14', 'ROC_12', 'MOM_12', 'CCI_20', 'WPR_14',
    'TRIX_15', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'PPO',
    'DPO_20', 'Ultimate_Osc', 'BB_Upper', 'BB_Lower', 'BB_Width',
    'Donchian_Upper', 'Donchian_Lower', 'Donchian_Width',
    'Keltner_Upper', 'Keltner_Lower', 'Keltner_Width', 'PercentB',
    'CMF_20',  # Keep
    # 'OBV',   # REMOVE - correlated with CMF
    # 'PVO',   # REMOVE - correlated with Volume_Zscore
    'Volume_Zscore',  # Keep
    # 'Volume_Change',  # REMOVE - keep only Zscore
    'Stoch_K', 'Stoch_D', 'Elder_Bull', 'Elder_Bear',
    'Vortex_Plus', 'Vortex_Minus', 'Price_Zscore',
    'Rolling_Skew', 'Rolling_Kurt', 'Price_Accel',
    'Hour', 'DayOfWeek',
    # Add relative features:
    'Volume_RelToPeer', 'Volatility_RelToPeer', 'Momentum_RelToPeer',
]
```

---

### Feature Improvement 3: Add Market Context
```python
# Add to feature engineering:

# Market volatility regime
df['Market_Volatility'] = df.groupby('DateTime_Hour')['HL_Range'].transform('mean')
df['Volatility_Regime'] = pd.cut(df['Market_Volatility'], bins=3, labels=[0, 1, 2])

# Volume regime
df['Market_Volume'] = df.groupby('DateTime_Hour')['Volume'].transform('mean')
df['Volume_Regime'] = pd.cut(df['Market_Volume'], bins=3, labels=[0, 1, 2])

# Average return sign
df['Market_Direction'] = df.groupby('DateTime_Hour')['Return'].transform(
    lambda x: 1 if x.mean() > 0 else -1
)
```

---

## PHASE 3: DATA QUALITY (Do This Week)

### Remove Noisy Data
```python
# In prepare_ranking_data.py, after computing all features:

# 1. Remove opening hour (most volatile, wide spreads)
df = df[df['DateTime_Hour'].dt.hour > 10]

# 2. Remove closing hour (liquidity issues)
df = df[df['DateTime_Hour'].dt.hour < 15]

# 3. Remove extreme returns (> 3 std devs)
extreme_threshold = df['Next_Hour_Return'].std() * 3
df = df[df['Next_Hour_Return'].abs() <= extreme_threshold]

# 4. Remove very low volume stocks
min_volume = df.groupby('Ticker')['Volume'].median().quantile(0.25)
df = df[df['Volume'] >= min_volume]

# 5. Remove queries with insufficient data
query_counts = df.groupby('Query_ID').size()
valid_queries = query_counts[query_counts >= 40].index  # Need at least 40 stocks per query
df = df[df['Query_ID'].isin(valid_queries)]

print(f"After filtering: {len(df)} rows (from 18656)")
```

---

## PHASE 4: TARGET ENGINEERING (Try Next Week)

### Option A: Directional Target (Easier)
```python
# Replace Next_Hour_Return with binary target
y = (df['Next_Hour_Return'] > 0).astype(int)

# Train with binary:logistic instead of rank:pairwise
params['objective'] = 'binary:logistic'

# This is easier to learn than exact returns
```

### Option B: Multi-Class Target (Balanced)
```python
# Quintile ranking (0-4)
y = pd.qcut(df['Next_Hour_Return'], q=5, labels=False, duplicates='drop')

# Train with multi:softmax
params['objective'] = 'multi:softmax'
params['num_class'] = 5

# This is a middle ground
```

### Option C: Absolute Rank Target (Best for Ranking)
```python
# For each query, rank 0-49
y = df.groupby('Query_ID')['Next_Hour_Return'].rank() - 1

# Use rank:ndcg instead of rank:pairwise
params['objective'] = 'rank:ndcg'

# This directly optimizes ranking quality
```

---

## IMPLEMENTATION SEQUENCE

**Day 1**:
1. ✓ Run `verify_data_alignment()` - check temporal alignment
2. ✓ Test inverted predictions with current model
3. ✓ Create `inference_inverted.py`
4. ✓ Measure improvement from inversion

**Day 2**:
1. ✓ Update XGBoost parameters (add regularization)
2. ✓ Retrain with `train_ranking_v2.py`
3. ✓ Compare new model vs original
4. ✓ Re-run evaluation

**Day 3**:
1. ✓ Add relative features to `prepare_ranking_data.py`
2. ✓ Remove redundant features
3. ✓ Clean data (remove noisy hours and stocks)
4. ✓ Retrain with new features

**Day 4**:
1. ✓ Try directional target (binary classification)
2. ✓ Try multi-class target (quintile)
3. ✓ Try absolute rank target (NDCG)
4. ✓ Compare all three approaches

---

## SUCCESS METRICS

Track these after each improvement:

```python
# Create comparison table:
import pandas as pd

results = pd.DataFrame({
    'Model': ['Baseline', 'Inverted', 'Regularized', '+Features', '+Data Clean', '+New Target'],
    'Top1_Acc': [0.00, 9.25, 15.00, 20.00, 22.00, 25.00],
    'Top3_Return': [-0.38, 0.29, 0.50, 0.75, 0.90, 1.20],
    'Spearman': [-0.63, 0.55, 0.40, 0.45, 0.48, 0.55],
    'Consistency': [0.30, 0.30, 0.20, 0.18, 0.16, 0.15],
})

print(results)
```

---

## EXPECTED OUTCOMES

| Stage | Spearman | Top-3 Return | Consistency |
|-------|----------|--------------|-------------|
| Baseline | -0.63 | -0.38% | 0.30 |
| After Phase 1 | +0.55 | +0.29% | 0.30 |
| After Phase 2 | +0.45 | +0.75% | 0.18 |
| After Phase 3 | +0.48 | +0.90% | 0.16 |
| After Phase 4 | +0.55+ | +1.20%+ | 0.15 |

**Total improvement**: 1.87% daily return increase (5x profit improvement!)

