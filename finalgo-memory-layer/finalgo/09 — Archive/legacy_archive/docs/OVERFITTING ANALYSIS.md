---
title: "Overfitting Analysis Report"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# Overfitting Analysis Report

## Executive Summary

**Issue**: Model shows significant overfitting with training Spearman correlation of 28.47% vs test correlation of 3.76% (7.6x gap).

**Root Causes Identified**:
1. Time-based data leakage in train/test split
2. Insufficient regularization in XGBoost
3. Low feature variance and signal-to-noise ratio
4. 48 features for only 47 stocks (feature-to-sample ratio issue)
5. Cumulative features creating temporal dependencies

---

## Data Analysis Results

### Dataset Overview
- **Total Samples**: 236,048 rows
- **Total Queries**: 5,048 hourly time periods
- **Stocks per Query**: 46.76 average (min: 22, max: 47)
- **Features**: 48 technical indicators
- **Time Period**: Dec 28, 2022 to Dec 4, 2025 (1,072 days / ~3 years)

### Target Variable Statistics
```
Next_Hour_Return:
  Mean:   0.0001 (0.01%)
  Std:    0.0063 (0.63%)
  Min:   -0.8003 (-80%)
  Max:    0.1335 (13.3%)
  25%:   -0.0022 (-0.22%)
  75%:    0.0022 (0.22%)
```

**Key Insight**: Returns are centered near zero with very small movements. This creates a **low signal-to-noise ratio** problem.

### Ticker Availability Issues
- **NBCC.NS**: Only 4,860 records (vs ~5,040 for others) - 3.6% less data
- **TATAMOTORS.NS**: Was in training list but not in realtime_trader (now removed)
- Most stocks have 5,000-5,042 records (consistent)

### Feature Variance Analysis

**Lowest Variance Features** (potential overfitting contributors):
```
OC_Range:       0.000029 (extremely low)
HL_Range:       0.000031 (extremely low)
Return:         0.000039 (extremely low)
Log_Return:     0.000048 (extremely low)
ROC_12:         0.000493 (very low)
```

**Implication**: These features have almost no variation across the dataset, providing minimal predictive signal but can still be overfit by the model.

---

## Root Cause #1: Time-Based Data Leakage

### Current Train/Test Split Strategy
```python
# From train_ranking.py line 55-56
unique_query_ids = df['Query_ID'].unique()
train_qids, test_qids = train_test_split(unique_query_ids, test_size=0.2, random_state=42)
```

**Problem**: `train_test_split` performs **random** splitting of queries, not temporal splitting.

### Why This Causes Overfitting

1. **Future Information Leakage**:
   - Training data from 2025 gets mixed with test data from 2023
   - Model learns patterns that appear consistent across random splits
   - But these patterns don't generalize to actual future data

2. **Autocorrelation Exploitation**:
   - Stock prices are autocorrelated (today's price predicts tomorrow's)
   - Random split allows model to "peek" at surrounding time periods
   - Test queries may be sandwiched between training queries

3. **Example**:
   ```
   Time:        T1    T2    T3    T4    T5    T6
   Random:     [TR] [TE] [TR] [TE] [TR] [TR]
   Temporal:   [TR] [TR] [TR] [TR] [TE] [TE]
   ```
   - Random split: Test T2 can learn from neighbors T1 and T3
   - Temporal split: Test T5 has no future information

**Evidence**:
- Training correlation: 28.47% (model fits random patterns well)
- Test correlation: 3.76% (fails to generalize temporally)
- 7.6x overfitting gap

### Solution
```python
# Temporal split instead of random split
df_sorted = df.sort_values('DateTime_Hour')
split_idx = int(len(unique_query_ids) * 0.8)
sorted_qids = sorted(unique_query_ids)
train_qids = sorted_qids[:split_idx]  # Earlier 80%
test_qids = sorted_qids[split_idx:]   # Recent 20%
```

**Expected Impact**: Reduce overfitting gap from 7.6x to 2-3x

---

## Root Cause #2: Insufficient Regularization

### Current XGBoost Parameters
```python
# From train_ranking.py line 89-97
params = {
    'objective': 'rank:pairwise',
    'eta': 0.1,                    # Learning rate (moderate)
    'max_depth': 6,                # Tree depth (allows complex patterns)
    'subsample': 0.8,              # Row sampling (some regularization)
    'colsample_bytree': 0.8,       # Column sampling (some regularization)
    'random_state': 42,
    'verbosity': 0,
}
num_boost_round=100
```

### Problems

1. **max_depth=6 is too high**:
   - With 48 features, depth-6 trees can memorize 2^6 = 64 different patterns
   - Allows model to overfit to training queries

2. **No min_child_weight**:
   - Trees can split on individual samples
   - Creates overly specific rules

3. **No gamma (min_split_loss)**:
   - No penalty for adding new splits
   - Encourages overfitting

4. **No L1/L2 regularization**:
   - No weight penalty on leaf values
   - Predictions can be extreme

5. **100 rounds may be too many**:
   - No early stopping
   - Model continues training past optimal point

### Recommended Parameters
```python
params = {
    'objective': 'rank:pairwise',
    'eta': 0.05,                   # Slower learning (was 0.1)
    'max_depth': 4,                # Shallower trees (was 6)
    'min_child_weight': 20,        # Require 20+ samples per leaf (was 1)
    'subsample': 0.7,              # More row sampling (was 0.8)
    'colsample_bytree': 0.7,       # More column sampling (was 0.8)
    'gamma': 2.0,                  # Minimum loss reduction for split
    'lambda': 10.0,                # L2 regularization on weights
    'alpha': 1.0,                  # L1 regularization on weights
    'random_state': 42,
    'verbosity': 1,
}

bst = xgb.train(params, dtrain,
                num_boost_round=500,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=50,  # Stop if no improvement
                verbose_eval=10)
```

**Expected Impact**: Reduce overfitting gap by 30-40%

---

## Root Cause #3: Low Signal-to-Noise Ratio

### Statistical Evidence

**Hourly Returns are Noisy**:
- Mean return: 0.01% (essentially zero)
- Std deviation: 0.63% (small movements)
- 50% of returns are between -0.22% and +0.22%

**Feature Variance Issues**:
- 5 of 48 features have variance < 0.0005
- These provide almost no discriminative power
- Model may overfit to noise in these features

### Why This Matters

1. **Prediction Difficulty**:
   - Hourly stock movements are inherently random
   - Technical indicators based on these movements are also noisy
   - Hard to find consistent ranking patterns

2. **Overfitting to Noise**:
   - Model finds spurious patterns in training data
   - These patterns are just noise, not true signal
   - Fail to generalize to test data

3. **Query Variance**:
   - Some hours have clear winners (high variance)
   - Some hours are essentially random (low variance)
   - Model struggles in low-variance periods

### Solutions

1. **Remove Low-Variance Features**:
   ```python
   # Remove features with variance < 0.001
   feature_variances = X_train.var(axis=0)
   keep_mask = feature_variances > 0.001
   X_train = X_train[:, keep_mask]
   ```

2. **Add Relative Features** (creates stronger ranking signal):
   ```python
   # For each query, compute relative position vs peers
   for col in feature_cols:
       df[f'{col}_Rank'] = df.groupby('Query_ID')[col].rank(pct=True)
       df[f'{col}_Zscore'] = df.groupby('Query_ID')[col].transform(
           lambda x: (x - x.mean()) / (x.std() + 1e-8)
       )
   ```

3. **Filter Low-Quality Queries**:
   ```python
   # Remove queries where all stocks moved < 0.1%
   query_ranges = df.groupby('Query_ID')['Next_Hour_Return'].apply(
       lambda x: x.max() - x.min()
   )
   valid_queries = query_ranges[query_ranges > 0.001].index
   df = df[df['Query_ID'].isin(valid_queries)]
   ```

**Expected Impact**: Improve test correlation by 50-100%

---

## Root Cause #4: Feature-to-Sample Ratio

### Current Situation
- **Features**: 48 technical indicators
- **Samples per query**: 46.76 average (47 stocks)
- **Ratio**: 48 features / 47 stocks = 1.02

**Problem**: With nearly as many features as samples per query, the model can memorize individual stocks rather than learn generalizable patterns.

### Mathematical Perspective

In ranking problems, we learn patterns **within each query**. With 47 stocks and 48 features:
- The feature space is 48-dimensional
- We only have ~47 points to learn from per query
- This is an **underdetermined system** prone to overfitting

**Analogy**: Fitting a 48-degree polynomial through 47 points - it will overfit!

### Solutions

1. **Feature Selection** (reduce from 48 to ~20):
   ```python
   # Keep only high-importance features
   from xgboost import plot_importance
   importance = bst.get_score(importance_type='gain')
   top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
   ```

2. **Dimensionality Reduction**:
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=20)
   X_reduced = pca.fit_transform(X)
   ```

3. **Feature Grouping**:
   - Combine correlated features into single indicators
   - Example: Average of SMA_6, SMA_12, EMA_12, EMA_24 → "Trend_Score"

**Expected Impact**: Reduce overfitting by 20-30%

---

## Root Cause #5: Cumulative Features with Temporal Dependencies

### Problematic Features

Several features accumulate over time and create temporal dependencies:

1. **OBV (On Balance Volume)**: `cumsum()` of volume
2. **Temporal features**: Hour, DayOfWeek

### Why This Causes Overfitting

**OBV Example**:
```python
def OBV(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()  # ← cumsum creates dependency!
```

- OBV value at time T depends on ALL previous times
- Training data OBV has context from 2022-2024
- Test data OBV has different baseline from training
- Model learns absolute OBV levels (train-specific) instead of OBV changes (generalizable)

**Hour/DayOfWeek Features**:
- Model might learn "Monday at 10am is always bullish"
- This pattern may hold in training data by chance
- But doesn't generalize to test data

### Solutions

1. **Replace OBV with OBV_Change**:
   ```python
   df['OBV'] = OBV(df)
   df['OBV_Change'] = df['OBV'].pct_change()  # Use rate of change
   ```

2. **Remove Temporal Features** (Hour, DayOfWeek):
   - These create dataset-specific patterns
   - Better to let model learn from price/volume patterns

3. **Use Differenced Features**:
   ```python
   # Instead of absolute levels, use changes
   df['Price_Diff'] = df['Close'].diff()
   df['Volume_Diff'] = df['Volume'].diff()
   ```

**Expected Impact**: Reduce overfitting by 10-15%

---

## Validation Metrics Comparison

### Current Performance
```
Training Spearman:     28.47%
Test Spearman:         3.76%
Overfitting Ratio:     7.6x
Full Eval Spearman:    23.48%
Top-1 Accuracy:        9.17%
MAE:                   4.01%
RMSE:                  5.43%
```

### Expected After Fixes

| Fix | Train ↓ | Test ↑ | Ratio ↓ |
|-----|---------|--------|---------|
| **Baseline** | 28.47% | 3.76% | 7.6x |
| + Temporal split | 25% | 8% | 3.1x |
| + Regularization | 20% | 12% | 1.7x |
| + Feature selection | 18% | 15% | 1.2x |
| + Remove cumulative | 17% | 16% | 1.1x |
| + Relative features | 22% | 20% | 1.1x |

**Target**: Test Spearman of 15-20% with overfitting ratio < 1.5x

---

## Implementation Priority

### Phase 1: Critical Fixes (High Impact, Easy)
1. ✅ **Remove TATAMOTORS.NS** - Done
2. **Implement temporal train/test split** - 30 min
3. **Add regularization parameters** - 15 min
4. **Add early stopping** - 5 min

**Expected Improvement**: Test correlation 3.76% → 12%

### Phase 2: Feature Engineering (High Impact, Moderate)
5. **Remove low-variance features** - 30 min
6. **Add relative ranking features** - 1 hour
7. **Replace OBV with OBV_Change** - 15 min
8. **Remove Hour/DayOfWeek features** - 5 min

**Expected Improvement**: Test correlation 12% → 18%

### Phase 3: Advanced (Moderate Impact, Hard)
9. **Feature selection (top 25 features)** - 1 hour
10. **Filter low-quality queries** - 30 min
11. **Cross-validation with time series split** - 1 hour

**Expected Improvement**: Test correlation 18% → 22%

---

## Recommended Code Changes

### 1. Update train_ranking.py - Temporal Split

**Replace lines 54-59**:
```python
# OLD - Random split (WRONG!)
unique_query_ids = df['Query_ID'].unique()
train_qids, test_qids = train_test_split(unique_query_ids, test_size=0.2, random_state=42)

# NEW - Temporal split (CORRECT!)
df['DateTime_Hour'] = pd.to_datetime(df['DateTime_Hour'])
df_sorted = df.sort_values('DateTime_Hour')
unique_query_ids_sorted = df_sorted['Query_ID'].unique()
split_idx = int(len(unique_query_ids_sorted) * 0.8)
train_qids = unique_query_ids_sorted[:split_idx]
test_qids = unique_query_ids_sorted[split_idx:]
```

### 2. Update train_ranking.py - Regularization

**Replace lines 89-101**:
```python
# OLD - Weak regularization
params = {
    'objective': 'rank:pairwise',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}
bst = xgb.train(params, dtrain, num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                verbose_eval=10)

# NEW - Strong regularization
params = {
    'objective': 'rank:pairwise',
    'eta': 0.05,
    'max_depth': 4,
    'min_child_weight': 20,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 2.0,
    'lambda': 10.0,
    'alpha': 1.0,
    'random_state': 42,
    'verbosity': 1,
}
bst = xgb.train(params, dtrain, num_boost_round=500,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=50,
                verbose_eval=10)
```

### 3. Update prepare_ranking_data.py - Remove OBV, Add Changes

**Replace lines 220-221**:
```python
# OLD
df['OBV'] = OBV(df)

# NEW
obv_temp = OBV(df)
df['OBV_Change'] = obv_temp.pct_change()  # Use rate of change instead
```

**Remove temporal features (lines 246-247)**:
```python
# REMOVE THESE LINES
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
```

### 4. Add Relative Features

**Add after line 248**:
```python
# Add relative ranking features (computed per query later)
# These will be added during ranking data creation after groupby
```

**Update lines 264-267**:
```python
# After creating Query_ID, add relative features
for col in ['RSI_14', 'MACD_Hist', 'CMF_20', 'Volume_Zscore', 'ROC_12']:
    if col in df_all.columns:
        df_all[f'{col}_RelRank'] = df_all.groupby('Query_ID')[col].rank(pct=True)
```

---

## Testing Plan

### 1. Baseline Test (Current Model)
```bash
python scripts/evaluate_model.py
# Record: Train=28.47%, Test=3.76%
```

### 2. After Temporal Split
```bash
# Modify train_ranking.py
python scripts/train_ranking.py
python scripts/evaluate_model.py
# Expected: Train=25%, Test=8%
```

### 3. After Regularization
```bash
# Modify params in train_ranking.py
python scripts/train_ranking.py
python scripts/evaluate_model.py
# Expected: Train=20%, Test=12%
```

### 4. After Feature Engineering
```bash
# Modify prepare_ranking_data.py
python scripts/prepare_ranking_data.py
python scripts/train_ranking.py
python scripts/evaluate_model.py
# Expected: Train=22%, Test=18%
```

### 5. Backtest Validation
```bash
# Test on Dec 1-4, 2024
python analysis/analyze_historical_day.py
# Compare returns before/after fixes
```

---

## Summary

### Key Findings

1. **Primary Cause**: Random train/test split causing temporal leakage (7.6x overfitting)
2. **Secondary Cause**: Weak regularization allowing memorization
3. **Tertiary Cause**: Low signal-to-noise ratio in hourly data
4. **Contributing Factors**: Cumulative features, low-variance features, too many features

### Quick Wins

- ✅ Remove TATAMOTORS.NS (Done)
- Implement temporal split (30 min, huge impact)
- Add regularization (15 min, high impact)
- Remove temporal features (5 min, moderate impact)

### Expected Outcome

With all fixes implemented:
- Test Spearman: 3.76% → 18-22%
- Overfitting ratio: 7.6x → 1.2-1.5x
- Top-1 accuracy: 9% → 15-20%
- Real-world backtesting: More consistent returns

### Next Steps

1. Implement Phase 1 fixes (temporal split + regularization)
2. Retrain model and validate improvement
3. If successful, proceed to Phase 2 (feature engineering)
4. Backtest on December 1-4 data
5. Deploy improved model to production
