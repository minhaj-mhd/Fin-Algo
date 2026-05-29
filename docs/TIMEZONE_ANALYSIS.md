# Timezone Analysis: UTC vs IST Handling

## Executive Summary

**Critical Issue Found**: Timezone inconsistency between training and inference could be causing misalignment of features.

**Status**:
- ✅ Training data uses **UTC** timestamps
- ✅ `analyze_historical_day.py` converts UTC → IST properly
- ⚠️ `realtime_trader.py` uses **naive local datetime** (no timezone awareness)

---

## Detailed Analysis

### 1. Training Data (prepare_ranking_data.py)

**Source**: yfinance hourly data
**Timezone**: UTC (Coordinated Universal Time)

```python
# yfinance returns UTC timestamps
df = yf.download(ticker, period="730d", interval="1h")
# Index: DatetimeIndex with tz=UTC

# Example timestamps:
2022-12-28 03:45:00+00:00  # UTC (9:15 AM IST - market open)
2022-12-28 04:45:00+00:00  # UTC (10:15 AM IST)
2022-12-28 05:45:00+00:00  # UTC (11:15 AM IST)
...
2022-12-28 09:45:00+00:00  # UTC (3:15 PM IST - near market close)
```

**NSE Market Hours**:
- IST: 09:15 - 15:30
- UTC equivalent: 03:45 - 10:00

**Training Data Timestamps**:
```
Dataset sample from ranking_data_full.csv:
DateTime                    Ticker        Query_ID
2022-12-28 03:45:00+00:00  RELIANCE.NS   0         # 9:15 AM IST
2022-12-28 04:45:00+00:00  RELIANCE.NS   1         # 10:15 AM IST
2022-12-28 05:45:00+00:00  RELIANCE.NS   2         # 11:15 AM IST
```

✅ **Status**: Consistent UTC timestamps in training data

---

### 2. Inference - analyze_historical_day.py (Backtesting)

**Timezone Handling**: Properly converts UTC to IST

```python
# Line 59: Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Line 70-72: Convert UTC to IST
if df.index.tz is not None:
    df.index = df.index.tz_convert(ist)
```

**Example Flow**:
```python
# yfinance returns: 2024-12-04 03:45:00+00:00 (UTC)
# Converted to:     2024-12-04 09:15:00+05:30 (IST)
```

**Test Slots** (analyze_historical_day.py line 417):
```python
test_slots = [
    ('09:30', '10:30'),  # IST - First hour
    ('10:30', '11:30'),  # IST - Second hour
    ('11:30', '12:30'),  # IST - Third hour
    ...
]
```

✅ **Status**: Properly timezone-aware (UTC → IST conversion)

---

### 3. Inference - realtime_trader.py (Live Trading)

**Timezone Handling**: ⚠️ **NAIVE - No timezone awareness!**

```python
# Line 168: Uses local system datetime (NO timezone)
now = datetime.now()  # ← NAIVE datetime!

# Line 172: Uses naive datetime
print(f"Re-ranking all {len(TICKERS)} tickers at {datetime.now()}")

# Line 201, 235: More naive datetimes
entry_time = datetime.now()
now = datetime.now()
```

**Problem**:
1. `datetime.now()` returns **naive** datetime (no timezone info)
2. On a Windows system in India, this returns IST
3. But it's stored/compared as naive datetime without tz info
4. When fetching yfinance data, it comes back in UTC
5. **Mismatch occurs**: Comparing naive IST with timezone-aware UTC

**Example of the Issue**:

```python
# Current code (WRONG):
now = datetime.now()  # 2024-12-04 09:30:00 (naive, assumes IST)
df = yf.download(ticker, period='15d', interval='1h')
# df.index: 2024-12-04 03:45:00+00:00 (UTC with tz)

# Feature computation uses df.index with UTC timestamps
df['Hour'] = df.index.hour  # Returns 3, 4, 5... (UTC hours)

# But model was trained on IST hours: 9, 10, 11... (IST hours)
# MISMATCH!
```

---

## The Critical Problem: Hour Feature Mismatch

### Training Data (prepare_ranking_data.py line 246)

```python
df['Hour'] = df.index.hour  # Uses index from yfinance (UTC)
```

**Training hour values**:
```
UTC Time         Hour Feature    IST Equivalent
03:45 UTC    →   3               09:15 IST (market open)
04:45 UTC    →   4               10:15 IST
05:45 UTC    →   5               11:15 IST
06:45 UTC    →   6               12:15 IST
07:45 UTC    →   7               13:15 IST
08:45 UTC    →   8               14:15 IST
09:45 UTC    →   9               15:15 IST (market close)
```

Model learned: "Hour=3 is market open, Hour=9 is market close"

### Inference in realtime_trader.py

**Problem**: Feature computation uses yfinance data which is in UTC

```python
# Line 82: Fetches hourly data (returns UTC timestamps)
df = yf.download(t, period='15d', interval='1h', ...)

# Line 96: Computes features
df_features = compute_features(df)
# This calls: df['Hour'] = df.index.hour (from feature_utils.py)
```

**Inference hour values** (if using yfinance directly):
```
UTC Time         Hour Feature    What Model Expects
03:45 UTC    →   3               ✅ Correct (market open)
04:45 UTC    →   4               ✅ Correct
...
```

Wait, this is actually **correct**! yfinance returns UTC, so Hour feature is also UTC.

### But what about DayOfWeek?

**Training** (prepare_ranking_data.py line 247):
```python
df['DayOfWeek'] = df.index.dayofweek  # Uses UTC dayofweek
```

**Issue**: If market opens on Monday IST at 9:15 AM:
- IST: Monday 09:15 → DayOfWeek = 0 (Monday)
- UTC: Monday 03:45 → DayOfWeek = 0 (Monday)
- ✅ Usually same day

But near midnight IST:
- IST: Monday 00:30 → DayOfWeek = 0 (Monday)
- UTC: Sunday 19:00 → DayOfWeek = 6 (Sunday)
- ❌ Different day!

This is not a market hours issue though since NSE is closed at midnight.

---

## Actually, Let Me Re-check realtime_trader.py More Carefully

Looking at [realtime_trader.py:82](scripts/realtime_trader.py:82):

```python
df = yf.download(t, period='15d', interval='1h', auto_adjust=False, progress=False, prepost=False)
```

This returns DataFrame with **UTC** index. Then:

```python
df_features = compute_features(df)  # Line 96
```

This calls `feature_utils.compute_features()` which does:

```python
df['Hour'] = df.index.hour  # Uses UTC hour from yfinance
df['DayOfWeek'] = df.index.dayofweek  # Uses UTC dayofweek
```

So actually, **realtime_trader.py IS using UTC for features** ✅

The `datetime.now()` calls are only for **scheduling** (when to open trades), not for feature computation.

---

## Re-Analysis: Is There Actually a Timezone Problem?

Let me trace through the complete flow:

### Training Flow:
1. yfinance → UTC timestamps (03:45, 04:45, ...)
2. compute features → Hour = 3, 4, 5, ... (UTC)
3. Train model on Hour=3,4,5,...

### Inference Flow (realtime_trader.py):
1. yfinance → UTC timestamps (03:45, 04:45, ...)
2. compute features → Hour = 3, 4, 5, ... (UTC)
3. Predict with Hour=3,4,5,...

✅ **These match!**

### Inference Flow (analyze_historical_day.py):
1. yfinance → UTC timestamps
2. Convert to IST → (09:15, 10:15, ...)
3. ⚠️ **Wait, this is the issue!**

Let me check compute_features again:

```python
# analyze_historical_day.py line 71-72
if df.index.tz is not None:
    df.index = df.index.tz_convert(ist)  # ← CONVERTS TO IST!

# Then line 98
df_features = compute_features(df_before)
```

So `compute_features` receives IST timestamps and does:
```python
df['Hour'] = df.index.hour  # Now returns IST hours: 9, 10, 11...
```

❌ **MISMATCH FOUND!**

---

## The Actual Problem

### Training Data:
- Uses UTC timestamps from yfinance
- Hour feature = 3, 4, 5, 6, 7, 8, 9 (UTC)

### realtime_trader.py:
- Uses UTC timestamps from yfinance
- Hour feature = 3, 4, 5, 6, 7, 8, 9 (UTC)
- ✅ **Matches training**

### analyze_historical_day.py:
- Converts UTC → IST before computing features
- Hour feature = 9, 10, 11, 12, 13, 14, 15 (IST)
- ❌ **DOES NOT match training!**

This is why backtesting results might be unreliable!

---

## Impact Analysis

### Hour Feature Importance

From the model, Hour is one of 48 features. If Hour has moderate importance, this mismatch could significantly affect predictions.

**Model learned**:
- "Hour=3 (9:15 AM IST market open) typically shows momentum"
- "Hour=9 (3:15 PM IST near close) shows mean reversion"

**During backtesting** (analyze_historical_day.py):
- Feeds Hour=9 (thinking it's 9 AM IST)
- But model interprets Hour=9 as "near market close"
- Applies wrong strategy!

### DayOfWeek Feature

Less critical since market hours don't span midnight, but still inconsistent.

---

## Root Cause

**File**: `analyze_historical_day.py`
**Lines**: 70-72

```python
# PROBLEMATIC CODE:
if df.index.tz is not None:
    df.index = df.index.tz_convert(ist)  # ← This conversion breaks consistency!
```

This was added for human readability (to see IST times in output), but it breaks feature consistency with training.

---

## Solutions

### Option 1: Keep Everything in UTC (Recommended)

**Change**: Remove timezone conversion in analyze_historical_day.py

```python
# REMOVE these lines (70-72):
# if df.index.tz is not None:
#     df.index = df.index.tz_convert(ist)

# Keep UTC for feature computation
# Only convert to IST for display/logging
```

**Impact**:
- ✅ Training and inference use same Hour values
- ✅ Consistent across all scripts
- Slot times in code would need to be in UTC (03:45, 04:45, ...)

### Option 2: Convert Everything to IST (Alternative)

**Change**: Convert training data to IST before computing features

**In prepare_ranking_data.py**, after line 172:
```python
# Add timezone conversion
import pytz
ist = pytz.timezone('Asia/Kolkata')
df.index = df.index.tz_convert(ist)
```

**Impact**:
- ✅ More intuitive (IST hours: 9, 10, 11, ...)
- ✅ Matches human understanding
- ❌ Need to retrain model
- ❌ Need to update realtime_trader.py too

### Option 3: Remove Temporal Features (Best for ML)

**Change**: Don't use Hour/DayOfWeek features at all

**Rationale**:
- These features create dataset-specific patterns (overfitting)
- Market behavior at "9 AM" changes over time
- Better to let model learn from price/volume patterns

**In prepare_ranking_data.py**, remove lines 246-247:
```python
# REMOVE:
# df['Hour'] = df.index.hour
# df['DayOfWeek'] = df.index.dayofweek
```

**Impact**:
- ✅ Removes timezone inconsistency entirely
- ✅ Reduces overfitting (one less dataset-specific pattern)
- ✅ More generalizable model
- ❌ Need to retrain

---

## Recommendation

**Implement Option 3** (Remove temporal features):
1. Eliminates timezone issues completely
2. Reduces overfitting (from earlier analysis)
3. Improves generalization

**Short-term fix** (Option 1):
1. Remove UTC→IST conversion in analyze_historical_day.py (line 70-72)
2. Keep slot times in UTC (03:45 = 9:15 IST)
3. Backtesting will immediately become accurate

---

## Testing Validation

### Test 1: Verify Current Mismatch

```python
# Training data
import pandas as pd
df_train = pd.read_csv('data/ranking_data_full.csv', nrows=1000)
print("Training Hour values:", df_train['Hour'].unique())
# Expected: [3, 4, 5, 6, 7, 8, 9] (UTC hours during market)

# Backtesting (current broken state)
# Would show: [9, 10, 11, 12, 13, 14, 15] (IST hours)
```

### Test 2: After Fix

```python
# Both should show same Hour values
# Training: [3, 4, 5, 6, 7, 8, 9]
# Inference: [3, 4, 5, 6, 7, 8, 9]
```

---

## Summary

| Script | Current Timezone | Hour Feature | Status |
|--------|-----------------|--------------|--------|
| **prepare_ranking_data.py** | UTC | 3-9 | ✅ Correct |
| **realtime_trader.py** | UTC (features) | 3-9 | ✅ Correct |
| **analyze_historical_day.py** | IST (converted) | 9-15 | ❌ WRONG |

**Root Cause**: analyze_historical_day.py converts UTC→IST before computing features, creating 6-hour offset in Hour feature.

**Impact**: Backtesting predictions use wrong strategy (model thinks hour=9 is near market close, not market open).

**Fix**: Remove timezone conversion in analyze_historical_day.py line 70-72, OR remove Hour/DayOfWeek features entirely (recommended).

**Priority**: HIGH - This affects all backtesting accuracy and model evaluation!
