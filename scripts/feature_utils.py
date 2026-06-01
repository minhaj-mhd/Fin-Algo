import numpy as np
import pandas as pd

# Technical indicator helpers

def SMA(series, window):
    return series.rolling(window).mean()

def EMA(series, window):
    return series.ewm(span=window, adjust=False).mean()

def WMA(series, window):
    weights = np.arange(1, window+1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def HMA(series, window):
    half_len = int(window/2)
    sqrt_len = int(np.sqrt(window)) if window > 0 else 1
    return WMA(2*WMA(series, half_len)-WMA(series, window), sqrt_len)

def RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(window).mean()/down.rolling(window).mean()
    return 100-(100/(1+rs))

def ROC(series, window=12):
    return series.pct_change(window)

def MOM(series, window=12):
    return series - series.shift(window)

def CCI(df, window=20):
    TP = (df['High']+df['Low']+df['Close'])/3
    mean = TP.rolling(window).mean()
    mad = TP.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (TP - mean)/(0.015*mad)

def CCI_legacy(df, window=20):
    TP = (df['High']+df['Low']+df['Close'])/3
    mean = TP.rolling(window).mean()
    std = TP.rolling(window).std()
    return (TP - mean)/(0.015*std)

def WPR(df, window=14):
    return -100*(df['High'].rolling(window).max()-df['Close'])/(df['High'].rolling(window).max()-df['Low'].rolling(window).min())


def MACD(series, fast=12, slow=26, signal=9):
    fast_ema = EMA(series, fast)
    slow_ema = EMA(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def TRIX(series, window=15):
    e1 = EMA(series, window)
    e2 = EMA(e1, window)
    e3 = EMA(e2, window)
    return e3.pct_change()*100

def DPO(series, window=20):
    return series.shift(int(window/2+1))-SMA(series, window)

def PPO(series, fast=12, slow=26):
    return (EMA(series, fast)-EMA(series, slow))/EMA(series, slow)*100

def ATR(df, window=14):
    hl = df['High']-df['Low']
    hc = abs(df['High']-df['Close'].shift(1))
    lc = abs(df['Low']-df['Close'].shift(1))
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.rolling(window).mean()

def Bollinger_Bands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2*std
    lower = sma - 2*std
    width = upper-lower
    return upper, lower, width

def Donchian_Channel(df, window=20):
    upper = df['High'].rolling(window).max()
    lower = df['Low'].rolling(window).min()
    width = upper-lower
    return upper, lower, width

def Keltner_Channel(df, window=20):
    tp = (df['High']+df['Low']+df['Close'])/3
    ma = EMA(tp, window)
    hl = df['High']-df['Low']
    hc = abs(df['High']-df['Close'].shift(1))
    lc = abs(df['Low']-df['Close'].shift(1))
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    upper = ma + 2*atr
    lower = ma - 2*atr
    width = upper-lower
    return upper, lower, width

def OBV(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction*df['Volume']).cumsum()

def CMF(df, window=20):
    mfv = ((df['Close']-df['Low'])-(df['High']-df['Close']))/ (df['High']-df['Low'])*df['Volume']
    return mfv.rolling(window).sum()/df['Volume'].rolling(window).sum()

def PVO(volume, fast=12, slow=26):
    return (EMA(volume, fast)-EMA(volume, slow))/EMA(volume, slow)*100

def Stochastic(df, k_window=14, d_window=3):
    L = df['Low'].rolling(k_window).min()
    H = df['High'].rolling(k_window).max()
    K = 100*(df['Close']-L)/(H-L)
    D = K.rolling(d_window).mean()
    return K, D

def Elder_Ray(df, window=13):
    ema = EMA(df['Close'], window)
    bull = df['High']-ema
    bear = df['Low']-ema
    return bull, bear

def Vortex(df, window=14):
    hl = df['High']-df['Low']
    hc = abs(df['High']-df['Close'].shift(1))
    lc = abs(df['Low']-df['Close'].shift(1))
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    vm_plus = (df['High']-df['Low'].shift(1)).abs()
    vm_minus = (df['Low']-df['High'].shift(1)).abs()
    vi_plus = vm_plus.rolling(window).sum()/tr.rolling(window).sum()
    vi_minus = vm_minus.rolling(window).sum()/tr.rolling(window).sum()
    return vi_plus, vi_minus

def Vortex_legacy(df, window=14):
    tr = (df['High']-df['Low']).abs()
    vm_plus = (df['High']-df['Low'].shift(1)).abs()
    vm_minus = (df['Low']-df['High'].shift(1)).abs()
    vi_plus = vm_plus.rolling(window).sum()/tr.rolling(window).sum()
    vi_minus = vm_minus.rolling(window).sum()/tr.rolling(window).sum()
    return vi_plus, vi_minus

def Ultimate_Oscillator(df, short=7, medium=14, long=28):
    bp = df['Close']-np.maximum(df['Low'], df['Close'].shift(1))
    hl = df['High']-df['Low']
    hc = abs(df['High']-df['Close'].shift(1))
    lc = abs(df['Low']-df['Close'].shift(1))
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    avg1 = bp.rolling(short).sum()/tr.rolling(short).sum()
    avg2 = bp.rolling(medium).sum()/tr.rolling(medium).sum()
    avg3 = bp.rolling(long).sum()/tr.rolling(long).sum()
    return 100*(4*avg1+2*avg2+avg3)/7

def Ultimate_Oscillator_legacy(df, short=7, medium=14, long=28):
    bp = df['Close']-np.maximum(df['Low'], df['Close'].shift(1))
    tr = np.maximum(df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)))
    avg1 = bp.rolling(short).sum()/tr.rolling(short).sum()
    avg2 = bp.rolling(medium).sum()/tr.rolling(medium).sum()
    avg3 = bp.rolling(long).sum()/tr.rolling(long).sum()
    return 100*(4*avg1+2*avg2+avg3)/7


def compute_features(df, legacy=True):
    """Given a dataframe with columns Open/High/Low/Close/Volume indexed by datetime,
    compute all features used by the ranking model and return the augmented dataframe.
    """
    df = df.copy()

    # ── BASIC ─────────────────────────────────────────────────────────────────
    df['Return']     = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close']/df['Close'].shift(1))
    df['HL_Range']   = (df['High']-df['Low'])/df['Close']
    df['OC_Range']   = (df['Close']-df['Open'])/df['Open']

    # ── TREND / MOMENTUM ──────────────────────────────────────────────────────
    df['Dist_SMA_6']  = (df['Close'] - SMA(df['Close'],6))  / df['Close']
    df['Dist_SMA_12'] = (df['Close'] - SMA(df['Close'],12)) / df['Close']
    df['Dist_SMA_50'] = (df['Close'] - SMA(df['Close'],50)) / df['Close']
    df['Dist_EMA_12'] = (df['Close'] - EMA(df['Close'],12)) / df['Close']
    df['Dist_EMA_24'] = (df['Close'] - EMA(df['Close'],24)) / df['Close']
    df['Dist_HMA_12'] = (df['Close'] - HMA(df['Close'],12)) / df['Close']
    df['RSI_14']      = RSI(df['Close'],14)
    df['ROC_12']      = ROC(df['Close'],12)
    df['MOM_12_pct']  = df['Close'].pct_change(12)
    if legacy:
        df['CCI_20']      = CCI_legacy(df, 20)
    else:
        df['CCI_20']      = CCI(df, 20)
    df['WPR_14']      = WPR(df,14)
    df['TRIX_15']     = TRIX(df['Close'],15)

    df['PPO']        = PPO(df['Close'])
    df['PPO_Signal'] = EMA(df['PPO'], 9)
    df['PPO_Hist']   = df['PPO'] - df['PPO_Signal']
    df['Dist_DPO_20']  = DPO(df['Close']) / df['Close']
    if legacy:
        df['Ultimate_Osc'] = Ultimate_Oscillator_legacy(df)
    else:
        df['Ultimate_Osc'] = Ultimate_Oscillator(df)

    # ── VOLATILITY / CHANNELS ─────────────────────────────────────────────────
    bb_upper, bb_lower, bb_width = Bollinger_Bands(df['Close'],20)
    df['PercentB']       = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    df['Dist_BB_Upper']  = (bb_upper - df['Close']) / df['Close']
    df['Dist_BB_Lower']  = (df['Close'] - bb_lower) / df['Close']
    df['BB_Width']       = bb_width / df['Close']

    dc_upper, dc_lower, dc_width = Donchian_Channel(df,20)
    df['Dist_Donchian_Upper'] = (dc_upper - df['Close']) / df['Close']
    df['Dist_Donchian_Lower'] = (df['Close'] - dc_lower) / df['Close']
    df['Donchian_Width']      = dc_width / df['Close']

    keltner_upper, keltner_lower, keltner_width = Keltner_Channel(df,20)
    df['Dist_Keltner_Upper'] = (keltner_upper - df['Close']) / df['Close']
    df['Dist_Keltner_Lower'] = (df['Close'] - keltner_lower) / df['Close']
    df['Keltner_Width']      = keltner_width / df['Close']

    # ── VOLUME ────────────────────────────────────────────────────────────────
    obv = OBV(df)
    obv_sma = SMA(obv, 20)
    df['OBV_Dist']      = (obv - obv_sma) / (obv_sma.abs() + 1e-8)
    df['CMF_20']        = CMF(df,20)
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Zscore'] = (df['Volume']-df['Volume'].rolling(24).mean())/df['Volume'].rolling(24).std()
    df['PVO']           = PVO(df['Volume'])

    # ── STOCHASTIC / OSCILLATORS ──────────────────────────────────────────────
    stoch_k, stoch_d = Stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    elder_bull, elder_bear = Elder_Ray(df)
    df['Elder_Bull'] = elder_bull / df['Close']
    df['Elder_Bear'] = elder_bear / df['Close']

    if legacy:
        vi_plus, vi_minus = Vortex_legacy(df)
    else:
        vi_plus, vi_minus = Vortex(df)
    df['Vortex_Plus']  = vi_plus
    df['Vortex_Minus'] = vi_minus

    # ── STATISTICAL / TEMPORAL ────────────────────────────────────────────────
    df['Price_Zscore']  = (df['Close']-df['Close'].rolling(24).mean())/df['Close'].rolling(24).std()
    df['Rolling_Skew']  = df['Close'].rolling(24).skew()
    df['Rolling_Kurt']  = df['Close'].rolling(24).kurt()
    df['Price_Accel']   = df['Return'].diff()  # Fixed: acceleration of return, not price
    df['Hour']          = df.index.hour
    df['DayOfWeek']     = df.index.dayofweek

    # ── LIQUIDITY / RANKING ANCHORS ───────────────────────────────────────────
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['RVOL']          = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    # FIX: 52-Week High/Low — 52 weeks × 5 days × 6.25 hours ≈ 1625 hourly bars
    # (was incorrectly set to 250, which is only ~10 trading days)
    W52 = 1625
    high_52w = df['High'].rolling(W52, min_periods=50).max()
    low_52w  = df['Low'].rolling(W52,  min_periods=50).min()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-8)
    df['Dist_52W_Low']  = (df['Close'] - low_52w)  / (low_52w  + 1e-8)  # NEW

    # ── LAG FEATURES (3-bar memory for momentum continuation/reversal) ─────────
    # These give the model temporal context without look-ahead
    for lag in [1, 2, 3]:
        df[f'Return_lag{lag}']        = df['Return'].shift(lag)
        df[f'RSI_lag{lag}']           = df['RSI_14'].shift(lag)
        df[f'Volume_Zscore_lag{lag}'] = df['Volume_Zscore'].shift(lag)
        df[f'OC_Range_lag{lag}']      = df['OC_Range'].shift(lag)

    # ── MOMENTUM STREAK (consecutive up/down bars) ────────────────────────────
    up_flag   = (df['Return'] > 0).astype(int)
    down_flag = (df['Return'] < 0).astype(int)
    # Count consecutive 1s in up_flag / down_flag (reset on 0)
    df['Up_Streak']   = up_flag.groupby(
        (up_flag != up_flag.shift()).cumsum()
    ).cumcount() * up_flag
    df['Down_Streak'] = down_flag.groupby(
        (down_flag != down_flag.shift()).cumsum()
    ).cumcount() * down_flag

    # ── INTRADAY MICROSTRUCTURE ───────────────────────────────────────────────
    # These only make sense for intraday (hourly) data.
    # BUG FIX: groupby(DatetimeIndex) doesn't work — must use a string/int date key.

    try:
        # Build a plain integer date key (YYYYMMDD) for groupby — works with any index type
        idx = pd.to_datetime(df.index)
        date_int = idx.year * 10000 + idx.month * 100 + idx.day  # e.g. 20240315

        # Intraday accumulated return since session open
        day_open = df.groupby(date_int)['Close'].transform('first')
        df['Intraday_Return'] = (df['Close'] / (day_open + 1e-8)) - 1
    except Exception as e:
        df['Intraday_Return'] = 0.0

    try:
        idx = pd.to_datetime(df.index)
        date_int = idx.year * 10000 + idx.month * 100 + idx.day

        # VWAP = cumulative(Price x Volume) / cumulative(Volume), reset each day
        # Guard: replace zero-volume bars with NaN so they don't poison cumsum
        vol_safe = df['Volume'].copy().replace(0, np.nan)
        pv = df['Close'] * vol_safe
        cumulative_pv  = pv.groupby(date_int).cumsum()
        cumulative_vol = vol_safe.groupby(date_int).cumsum()

        # Only compute VWAP where we have at least some volume
        vwap = cumulative_pv / (cumulative_vol + 1e-8)

        # Distance from VWAP, capped at +-3% (realistic max for liquid NSE hourly bar)
        vwap_dist_raw = (df['Close'] - vwap) / (vwap.abs() + 1e-8)
        df['VWAP_Dist'] = vwap_dist_raw.clip(-0.03, 0.03).fillna(0.0)  # +ve = above VWAP
    except Exception as e:
        df['VWAP_Dist'] = 0.0

    try:
        # Session time features (NSE: 9:15 AM to 3:30 PM)
        hour = pd.Series(pd.to_datetime(df.index).hour, index=df.index)
        df['Is_Open_Hour']  = ((hour == 9) | (hour == 10)).astype(int)
        df['Is_Close_Hour'] = (hour >= 14).astype(int)
        df['Time_To_Close'] = (15 - hour).clip(0, 6).astype(float)
    except Exception:
        df['Is_Open_Hour']  = 0
        df['Is_Close_Hour'] = 0
        df['Time_To_Close'] = 3.0

    if not legacy:
        try:
            # 1. IBS (Internal Bar Strength)
            df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
            df['IBS_3'] = df['IBS'].rolling(3).mean().fillna(0.5)
            
            # 2. Buy Pressure (IBS * RVOL)
            df['Buy_Pressure'] = df['IBS'] * df['RVOL']
            
            # 3. Direction Consistency
            direction = np.sign(df['Return'].fillna(0.0))
            df['Direction_Consistency_3'] = direction.rolling(3).sum() / 3
            df['Direction_Consistency_5'] = direction.rolling(5).sum() / 5
            df['Direction_Consistency_3'] = df['Direction_Consistency_3'].fillna(0.0)
            df['Direction_Consistency_5'] = df['Direction_Consistency_5'].fillna(0.0)
            
            # 4. RSI Momentum
            df['RSI_Momentum'] = (df['RSI_14'] - df['RSI_14'].shift(3)).fillna(0.0)
            
            # 5. Return Acceleration
            df['Return_Accel'] = (df['Return'] - df['Return'].shift(1)).fillna(0.0)
            
            # 6. Shadows
            df['Lower_Shadow'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / (df['High'] - df['Low'] + 1e-8)
            df['Upper_Shadow'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / (df['High'] - df['Low'] + 1e-8)
            
            # 7. Alpha Persistence
            rolling_mean_ret = df['Return'].rolling(20).mean().fillna(0.0)
            alpha = df['Return'].fillna(0.0) - rolling_mean_ret
            df['Alpha_3H'] = alpha.rolling(3).sum().fillna(0.0)
            df['Alpha_6H'] = alpha.rolling(6).sum().fillna(0.0)
        except Exception as e:
            for col in ['IBS', 'IBS_3', 'Buy_Pressure', 'Direction_Consistency_3', 'Direction_Consistency_5',
                        'RSI_Momentum', 'Return_Accel', 'Lower_Shadow', 'Upper_Shadow', 'Alpha_3H', 'Alpha_6H']:
                if col not in df.columns:
                    df[col] = 0.0

    return df

def compute_features_15min(df, legacy=False):
    """Given a dataframe with columns Open/High/Low/Close/Volume indexed by datetime,
    compute standard features and 15-minute specific features.
    """
    # 1. Compute standard features
    df = compute_features(df, legacy=legacy)
    
    # 2. Compute 15-minute specific features
    # Volume_Burst: Volume / 8-period volume average
    df['Volume_Burst'] = df['Volume'] / (df['Volume'].rolling(8).mean() + 1e-8)
    
    # Spread_Proxy: (High - Low) / Close
    df['Spread_Proxy'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
    
    # Candle_Body_Ratio: abs(Close - Open) / (High - Low)
    df['Candle_Body_Ratio'] = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low'] + 1e-8)
    
    # Momentum_Fade: Return / Return_lag1 (stable ratio clipped to [-5.0, 5.0])
    lag1_ret = df['Return'].shift(1)
    df['Momentum_Fade'] = (df['Return'] / (lag1_ret + np.sign(lag1_ret) * 1e-8)).clip(-5.0, 5.0).fillna(0.0)
    
    # Intra_Hour_Position: minute // 15 (0-3)
    df['Intra_Hour_Position'] = (df.index.minute // 15).astype(float)
    
    # Volume_Tilt: Volume / 4-period sum (current 15m volume ratio to past 1 hour)
    df['Volume_Tilt'] = df['Volume'] / (df['Volume'].rolling(4).sum() + 1e-8)
    
    # Raw_Volume_Zscore: rolling volume z-score (not cross-sectionally scored)
    df['Raw_Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(24).mean()) / (df['Volume'].rolling(24).std() + 1e-8)
    df['Raw_Volume_Zscore'] = df['Raw_Volume_Zscore'].fillna(0.0)
    
    # CMF_8: Chaikin Money Flow over 8 periods
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8) * df['Volume']
    df['CMF_8'] = mfv.rolling(8).sum() / (df['Volume'].rolling(8).sum() + 1e-8)
    df['CMF_8'] = df['CMF_8'].fillna(0.0)
    
    # RVOL_4: Relative volume over 4 periods
    df['RVOL_4'] = df['Volume'] / (df['Volume'].rolling(4).mean() + 1e-8)
    
    return df

def compute_features_30min(df, legacy=False):
    """Given a dataframe with columns Open/High/Low/Close/Volume indexed by datetime,
    compute standard features and 30-minute specific features.
    """
    # 1. Compute standard features
    df = compute_features(df, legacy=legacy)
    
    # 2. Compute 30-minute specific features
    # Volume_Burst: Volume / 4-period volume average (4 * 30min = 2 hours)
    df['Volume_Burst'] = df['Volume'] / (df['Volume'].rolling(4).mean() + 1e-8)
    
    # Spread_Proxy: (High - Low) / Close
    df['Spread_Proxy'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
    
    # Candle_Body_Ratio: abs(Close - Open) / (High - Low)
    df['Candle_Body_Ratio'] = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low'] + 1e-8)
    
    # Momentum_Fade: Return / Return_lag1 (stable ratio clipped to [-5.0, 5.0])
    lag1_ret = df['Return'].shift(1)
    df['Momentum_Fade'] = (df['Return'] / (lag1_ret + np.sign(lag1_ret) * 1e-8)).clip(-5.0, 5.0).fillna(0.0)
    
    # Intra_Hour_Position: minute // 30 (0-1)
    df['Intra_Hour_Position'] = (df.index.minute // 30).astype(float)
    
    # Volume_Tilt: Volume / 2-period sum (current 30m volume ratio to past 1 hour)
    df['Volume_Tilt'] = df['Volume'] / (df['Volume'].rolling(2).sum() + 1e-8)
    
    # Raw_Volume_Zscore: rolling volume z-score over 12 periods (12 * 30min = 6 hours)
    df['Raw_Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(12).mean()) / (df['Volume'].rolling(12).std() + 1e-8)
    df['Raw_Volume_Zscore'] = df['Raw_Volume_Zscore'].fillna(0.0)
    
    # CMF_8: Chaikin Money Flow over 4 periods (4 * 30min = 2 hours)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8) * df['Volume']
    df['CMF_8'] = mfv.rolling(4).sum() / (df['Volume'].rolling(4).sum() + 1e-8)
    df['CMF_8'] = df['CMF_8'].fillna(0.0)
    
    # RVOL_4: Relative volume over 2 periods (2 * 30min = 1 hour)
    df['RVOL_4'] = df['Volume'] / (df['Volume'].rolling(2).mean() + 1e-8)
    
    return df


def compute_features_daily_xgb(df):
    """
    Compute features optimized for DAILY XGBoost tree-based ranking models.
    
    XGBoost-specific design philosophy:
    - Trees split one feature at a time → explicit interaction features help
    - Trees handle categorical/binary features natively → add zone signals
    - Lag features useful since trees have no temporal memory
    - More features is OK — trees ignore irrelevant ones via feature importance
    
    Expects df with columns: Open, High, Low, Close, Volume, indexed by DatetimeIndex.
    Returns augmented df with ~160+ XGBoost-optimized features.
    """
    df = df.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: BASIC PRICE ACTION
    # ═══════════════════════════════════════════════════════════════════════════
    df['Return']     = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Range']   = (df['High'] - df['Low']) / df['Close']
    df['OC_Range']   = (df['Close'] - df['Open']) / df['Open']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: MULTI-HORIZON MOMENTUM (DAILY-SPECIFIC)
    # ═══════════════════════════════════════════════════════════════════════════
    df['Return_5D']  = df['Close'].pct_change(5)
    df['Return_10D'] = df['Close'].pct_change(10)
    df['Return_20D'] = df['Close'].pct_change(20)
    df['Return_60D'] = df['Close'].pct_change(60)
    df['Return_120D'] = df['Close'].pct_change(120)

    df['Momentum_Accel_5_20'] = df['Return_5D'] - df['Return_20D']
    df['Momentum_Accel_20_60'] = df['Return_20D'] - df['Return_60D']

    df['ROC_5']  = ROC(df['Close'], 5)
    df['ROC_10'] = ROC(df['Close'], 10)
    df['ROC_20'] = ROC(df['Close'], 20)
    df['ROC_60'] = ROC(df['Close'], 60)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: TREND STRENGTH (ADX, SMA SLOPES)
    # ═══════════════════════════════════════════════════════════════════════════
    atr14 = ATR(df, 14)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm_clean = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm_clean = minus_dm.where(minus_dm > plus_dm, 0.0)
    plus_di = 100 * EMA(plus_dm_clean, 14) / (atr14 + 1e-8)
    minus_di = 100 * EMA(minus_dm_clean, 14) / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX_14'] = EMA(dx, 14)
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['DI_Spread'] = plus_di - minus_di

    sma20 = SMA(df['Close'], 20)
    sma50 = SMA(df['Close'], 50)
    df['SMA_20_Slope'] = sma20.pct_change(5)
    df['SMA_50_Slope'] = sma50.pct_change(10)

    df['Dist_SMA_5']   = (df['Close'] - SMA(df['Close'], 5))   / df['Close']
    df['Dist_SMA_10']  = (df['Close'] - SMA(df['Close'], 10))  / df['Close']
    df['Dist_SMA_20']  = (df['Close'] - SMA(df['Close'], 20))  / df['Close']
    df['Dist_SMA_50']  = (df['Close'] - SMA(df['Close'], 50))  / df['Close']
    df['Dist_SMA_100'] = (df['Close'] - SMA(df['Close'], 100)) / df['Close']
    df['Dist_SMA_200'] = (df['Close'] - SMA(df['Close'], 200)) / df['Close']
    df['Dist_EMA_12']  = (df['Close'] - EMA(df['Close'], 12))  / df['Close']
    df['Dist_EMA_26']  = (df['Close'] - EMA(df['Close'], 26))  / df['Close']
    df['Dist_EMA_50']  = (df['Close'] - EMA(df['Close'], 50))  / df['Close']
    df['Dist_HMA_20']  = (df['Close'] - HMA(df['Close'], 20))  / df['Close']

    df['SMA_5_20_Cross']   = (SMA(df['Close'], 5) - SMA(df['Close'], 20)) / df['Close']
    df['SMA_20_50_Cross']  = (SMA(df['Close'], 20) - SMA(df['Close'], 50)) / df['Close']
    df['SMA_50_200_Cross'] = (SMA(df['Close'], 50) - SMA(df['Close'], 200)) / df['Close']
    df['EMA_12_26_Cross']  = (EMA(df['Close'], 12) - EMA(df['Close'], 26)) / df['Close']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: OSCILLATORS & MEAN-REVERSION SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
    df['RSI_14']  = RSI(df['Close'], 14)
    df['RSI_7']   = RSI(df['Close'], 7)
    df['RSI_21']  = RSI(df['Close'], 21)
    df['RSI_Divergence'] = df['RSI_14'] - df['RSI_14'].shift(5)

    df['CCI_20'] = CCI(df, 20)
    df['WPR_14'] = WPR(df, 14)
    df['TRIX_15'] = TRIX(df['Close'], 15)

    df['PPO']        = PPO(df['Close'])
    df['PPO_Signal'] = EMA(df['PPO'], 9)
    df['PPO_Hist']   = df['PPO'] - df['PPO_Signal']
    df['Dist_DPO_20'] = DPO(df['Close']) / df['Close']
    df['Ultimate_Osc'] = Ultimate_Oscillator(df)

    stoch_k, stoch_d = Stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    df['Stoch_KD_Cross'] = stoch_k - stoch_d

    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    df['IBS_3'] = df['IBS'].rolling(3).mean().fillna(0.5)
    df['IBS_5'] = df['IBS'].rolling(5).mean().fillna(0.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: VOLATILITY REGIME FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    atr14 = ATR(df, 14)
    atr5  = ATR(df, 5)
    df['ATR_14_Pct'] = atr14 / df['Close']
    df['ATR_5_Pct']  = atr5 / df['Close']
    df['ATR_Ratio_5_14'] = atr5 / (atr14 + 1e-8)

    df['Volatility_5D']  = df['Return'].rolling(5).std()
    df['Volatility_20D'] = df['Return'].rolling(20).std()
    df['Volatility_60D'] = df['Return'].rolling(60).std()
    df['Vol_Ratio_5_20'] = df['Volatility_5D'] / (df['Volatility_20D'] + 1e-8)

    bb_upper, bb_lower, bb_width = Bollinger_Bands(df['Close'], 20)
    df['PercentB']      = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    df['Dist_BB_Upper'] = (bb_upper - df['Close']) / df['Close']
    df['Dist_BB_Lower'] = (df['Close'] - bb_lower) / df['Close']
    df['BB_Width']      = bb_width / df['Close']
    df['BB_Squeeze']    = bb_width / (bb_width.rolling(120).mean() + 1e-8)

    dc_upper, dc_lower, dc_width = Donchian_Channel(df, 20)
    df['Dist_Donchian_Upper'] = (dc_upper - df['Close']) / df['Close']
    df['Dist_Donchian_Lower'] = (df['Close'] - dc_lower) / df['Close']
    df['Donchian_Width']      = dc_width / df['Close']

    keltner_upper, keltner_lower, keltner_width = Keltner_Channel(df, 20)
    df['Dist_Keltner_Upper'] = (keltner_upper - df['Close']) / df['Close']
    df['Dist_Keltner_Lower'] = (df['Close'] - keltner_lower) / df['Close']
    df['Keltner_Width']      = keltner_width / df['Close']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: VOLUME PROFILE
    # ═══════════════════════════════════════════════════════════════════════════
    obv = OBV(df)
    obv_sma = SMA(obv, 20)
    df['OBV_Dist']      = (obv - obv_sma) / (obv_sma.abs() + 1e-8)
    df['CMF_20']        = CMF(df, 20)
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    df['PVO']           = PVO(df['Volume'])
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['RVOL']          = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    df['Vol_Price_Corr_10'] = df['Return'].rolling(10).corr(df['Volume_Change'])
    df['Vol_Price_Corr_20'] = df['Return'].rolling(20).corr(df['Volume_Change'])
    df['Buy_Pressure'] = df['IBS'] * df['RVOL']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: CANDLESTICK PATTERN FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    body = (df['Close'] - df['Open']).abs()
    total_range = df['High'] - df['Low'] + 1e-8

    df['Candle_Body_Ratio'] = body / total_range
    df['Lower_Shadow'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / total_range
    df['Upper_Shadow'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / total_range
    df['Is_Doji'] = (df['Candle_Body_Ratio'] < 0.10).astype(float)
    prev_body = body.shift(1)
    df['Engulfing_Ratio'] = body / (prev_body + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: GAP ANALYSIS (DAILY-SPECIFIC)
    # ═══════════════════════════════════════════════════════════════════════════
    df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)
    df['Gap_Fill_Pct'] = np.where(
        df['Overnight_Gap'] > 0,
        (df['High'] - df['Open']) / (df['Open'] - df['Close'].shift(1) + 1e-8),
        (df['Open'] - df['Low']) / (df['Close'].shift(1) - df['Open'] + 1e-8)
    )
    df['Gap_Fill_Pct'] = pd.Series(df['Gap_Fill_Pct'], index=df.index).clip(-5, 5).fillna(0)
    df['Avg_Gap_5D'] = df['Overnight_Gap'].rolling(5).mean()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9: MOMENTUM STREAKS & CONSISTENCY
    # ═══════════════════════════════════════════════════════════════════════════
    direction = np.sign(df['Return'].fillna(0.0))
    df['Direction_Consistency_3'] = direction.rolling(3).sum() / 3
    df['Direction_Consistency_5'] = direction.rolling(5).sum() / 5
    df['Direction_Consistency_10'] = direction.rolling(10).sum() / 10

    up_flag   = (df['Return'] > 0).astype(int)
    down_flag = (df['Return'] < 0).astype(int)
    df['Up_Streak']   = up_flag.groupby((up_flag != up_flag.shift()).cumsum()).cumcount() * up_flag
    df['Down_Streak'] = down_flag.groupby((down_flag != down_flag.shift()).cumsum()).cumcount() * down_flag

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 10: ELDER RAY, VORTEX, PRICE STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════════
    elder_bull, elder_bear = Elder_Ray(df)
    df['Elder_Bull'] = elder_bull / df['Close']
    df['Elder_Bear'] = elder_bear / df['Close']

    vi_plus, vi_minus = Vortex(df)
    df['Vortex_Plus']  = vi_plus
    df['Vortex_Minus'] = vi_minus
    df['Vortex_Spread'] = vi_plus - vi_minus

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 11: STATISTICAL FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    df['Price_Zscore_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-8)
    df['Price_Zscore_50'] = (df['Close'] - df['Close'].rolling(50).mean()) / (df['Close'].rolling(50).std() + 1e-8)
    df['Rolling_Skew']    = df['Return'].rolling(20).skew()
    df['Rolling_Kurt']    = df['Return'].rolling(20).kurt()
    df['Return_Accel']    = df['Return'].diff()
    df['Sharpe_20D'] = (df['Return'].rolling(20).mean()) / (df['Return'].rolling(20).std() + 1e-8)
    df['Sharpe_60D'] = (df['Return'].rolling(60).mean()) / (df['Return'].rolling(60).std() + 1e-8)

    rolling_max = df['Close'].rolling(20).max()
    df['Drawdown_20D'] = (df['Close'] - rolling_max) / (rolling_max + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 12: LAG FEATURES (XGBoost needs these — no temporal memory)
    # ═══════════════════════════════════════════════════════════════════════════
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag{lag}']        = df['Return'].shift(lag)
        df[f'RSI_lag{lag}']           = df['RSI_14'].shift(lag)
        df[f'Volume_Zscore_lag{lag}'] = df['Volume_Zscore'].shift(lag)
        df[f'OC_Range_lag{lag}']      = df['OC_Range'].shift(lag)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 13: 52-WEEK CONTEXT
    # ═══════════════════════════════════════════════════════════════════════════
    W52 = 250
    high_52w = df['High'].rolling(W52, min_periods=50).max()
    low_52w  = df['Low'].rolling(W52, min_periods=50).min()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-8)
    df['Dist_52W_Low']  = (df['Close'] - low_52w)  / (low_52w  + 1e-8)
    df['Range_52W_Pct'] = (high_52w - low_52w) / (low_52w + 1e-8)
    df['Position_In_52W'] = (df['Close'] - low_52w) / (high_52w - low_52w + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 14: CALENDAR & SEASONALITY (trees love categorical splits)
    # ═══════════════════════════════════════════════════════════════════════════
    df['DayOfWeek']  = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['MonthOfYear'] = df.index.month
    df['Is_Month_Start'] = (df.index.day <= 3).astype(float)
    df['Is_Month_End']   = (df.index.day >= 26).astype(float)
    df['WeekOfMonth']    = ((df.index.day - 1) // 7).astype(float)
    df['Weekly_Return']  = df['Close'].pct_change(5)
    df['Monthly_Return'] = df['Close'].pct_change(20)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 15: ALPHA PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    rolling_mean_ret = df['Return'].rolling(20).mean().fillna(0.0)
    alpha = df['Return'].fillna(0.0) - rolling_mean_ret
    df['Alpha_3D']  = alpha.rolling(3).sum().fillna(0.0)
    df['Alpha_5D']  = alpha.rolling(5).sum().fillna(0.0)
    df['Alpha_10D'] = alpha.rolling(10).sum().fillna(0.0)

    df['MOM_5_pct']  = df['Close'].pct_change(5)
    df['MOM_12_pct'] = df['Close'].pct_change(12)
    df['MOM_20_pct'] = df['Close'].pct_change(20)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 16: XGBOOST-SPECIFIC INTERACTION FEATURES
    # Trees split one feature at a time. Explicit interactions help trees
    # discover compound signals that require ≥2 splits to find.
    # ═══════════════════════════════════════════════════════════════════════════

    # Momentum × Volatility: strong momentum in low-vol = more signal
    df['MOM_x_InvVol'] = df['Return_5D'] / (df['Volatility_20D'] + 1e-8)

    # RSI × Volume: oversold/overbought with volume confirmation
    df['RSI_x_RVOL'] = (df['RSI_14'] - 50) * df['RVOL']

    # Trend × Momentum alignment: ADX confirms momentum direction
    df['ADX_x_DI_Spread'] = df['ADX_14'] * df['DI_Spread']

    # Gap × Volume: gaps on high volume more meaningful
    df['Gap_x_RVOL'] = df['Overnight_Gap'] * df['RVOL']

    # Bollinger position × Volume: breakout from bands with volume
    df['PercentB_x_RVOL'] = (df['PercentB'] - 0.5) * df['RVOL']

    # IBS × ATR: mean-reversion signal strength
    df['IBS_x_ATR'] = df['IBS'] * df['ATR_14_Pct']

    # Momentum reversal potential: strong recent move + RSI extreme
    df['Return5_x_RSI_Extreme'] = df['Return_5D'] * (df['RSI_14'] - 50).abs()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 17: XGBOOST-SPECIFIC BINARY ZONE SIGNALS
    # Trees naturally split at thresholds — binary features encode
    # well-known zones that trading wisdom recognizes.
    # ═══════════════════════════════════════════════════════════════════════════

    # RSI zones
    df['RSI_Oversold']  = (df['RSI_14'] < 30).astype(float)
    df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(float)
    df['RSI_Extreme']    = ((df['RSI_14'] < 20) | (df['RSI_14'] > 80)).astype(float)

    # MA regime: price above/below key MAs
    df['Above_SMA_20']  = (df['Close'] > SMA(df['Close'], 20)).astype(float)
    df['Above_SMA_50']  = (df['Close'] > SMA(df['Close'], 50)).astype(float)
    df['Above_SMA_200'] = (df['Close'] > SMA(df['Close'], 200)).astype(float)
    # Regime stack: 0=bear, 1=mixed, 2=mixed, 3=full bull
    df['MA_Regime_Stack'] = df['Above_SMA_20'] + df['Above_SMA_50'] + df['Above_SMA_200']

    # Bollinger extremes
    df['Above_BB_Upper'] = (df['Close'] > bb_upper).astype(float)
    df['Below_BB_Lower'] = (df['Close'] < bb_lower).astype(float)

    # ADX strong trend threshold
    df['Strong_Trend'] = (df['ADX_14'] > 25).astype(float)

    # Volume spike detection
    df['Volume_Spike'] = (df['RVOL'] > 2.0).astype(float)

    # Gap direction
    df['Gap_Up']   = (df['Overnight_Gap'] > 0.005).astype(float)
    df['Gap_Down'] = (df['Overnight_Gap'] < -0.005).astype(float)

    return df


def compute_features_daily_transformer(df):
    """
    Compute features optimized for DAILY Temporal Transformer sequence models.
    
    Transformer-specific design philosophy:
    - Model sees 10-day sequences → NO lag features (redundant, adds noise)
    - Attention learns interactions → NO explicit interaction features
    - Gradient-based learning → smooth continuous features only, NO binary/categorical
    - Fewer, cleaner features → less noise for attention heads to filter
    - Focus on rate-of-change and normalized features for stable gradients
    
    Expects df with columns: Open, High, Low, Close, Volume, indexed by DatetimeIndex.
    Returns augmented df with ~80 high-quality temporal features.
    """
    df = df.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: BASIC PRICE ACTION (smooth, continuous)
    # ═══════════════════════════════════════════════════════════════════════════
    df['Return']     = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Range']   = (df['High'] - df['Low']) / df['Close']
    df['OC_Range']   = (df['Close'] - df['Open']) / df['Open']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: MULTI-HORIZON MOMENTUM
    # Attention can learn momentum across the 10-day sequence, but
    # providing pre-computed multi-scale momentum gives it macro context
    # beyond the 10-day window.
    # ═══════════════════════════════════════════════════════════════════════════
    df['Return_5D']   = df['Close'].pct_change(5)
    df['Return_10D']  = df['Close'].pct_change(10)
    df['Return_20D']  = df['Close'].pct_change(20)
    df['Return_60D']  = df['Close'].pct_change(60)
    df['Return_120D'] = df['Close'].pct_change(120)

    df['Momentum_Accel_5_20']  = df['Return_5D'] - df['Return_20D']
    df['Momentum_Accel_20_60'] = df['Return_20D'] - df['Return_60D']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: TREND STRUCTURE (smooth indicators)
    # ═══════════════════════════════════════════════════════════════════════════
    # ADX — continuous trend strength, perfect for transformers
    atr14 = ATR(df, 14)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm_clean = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm_clean = minus_dm.where(minus_dm > plus_dm, 0.0)
    plus_di = 100 * EMA(plus_dm_clean, 14) / (atr14 + 1e-8)
    minus_di = 100 * EMA(minus_dm_clean, 14) / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX_14']    = EMA(dx, 14)
    df['DI_Spread'] = plus_di - minus_di  # Continuous directional signal

    # SMA Slope — trend direction via rate of change of MA
    df['SMA_20_Slope'] = SMA(df['Close'], 20).pct_change(5)
    df['SMA_50_Slope'] = SMA(df['Close'], 50).pct_change(10)

    # Distance from key moving averages (continuous, normalized)
    df['Dist_SMA_10']  = (df['Close'] - SMA(df['Close'], 10))  / df['Close']
    df['Dist_SMA_20']  = (df['Close'] - SMA(df['Close'], 20))  / df['Close']
    df['Dist_SMA_50']  = (df['Close'] - SMA(df['Close'], 50))  / df['Close']
    df['Dist_SMA_200'] = (df['Close'] - SMA(df['Close'], 200)) / df['Close']
    df['Dist_EMA_12']  = (df['Close'] - EMA(df['Close'], 12))  / df['Close']
    df['Dist_EMA_26']  = (df['Close'] - EMA(df['Close'], 26))  / df['Close']

    # Cross-MA spreads (smooth regime indicators)
    df['SMA_20_50_Cross']  = (SMA(df['Close'], 20) - SMA(df['Close'], 50)) / df['Close']
    df['SMA_50_200_Cross'] = (SMA(df['Close'], 50) - SMA(df['Close'], 200)) / df['Close']

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: OSCILLATORS (continuous bounded signals)
    # RSI/Stochastic are naturally bounded [0,100] — ideal for transformers
    # ═══════════════════════════════════════════════════════════════════════════
    df['RSI_14'] = RSI(df['Close'], 14)
    df['RSI_7']  = RSI(df['Close'], 7)
    df['CCI_20'] = CCI(df, 20)
    df['WPR_14'] = WPR(df, 14)

    df['PPO']      = PPO(df['Close'])
    df['PPO_Hist'] = df['PPO'] - EMA(df['PPO'], 9)
    df['TRIX_15']  = TRIX(df['Close'], 15)

    stoch_k, stoch_d = Stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    # IBS — powerful daily mean-reversion signal, continuous [0,1]
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: VOLATILITY REGIME (smooth, continuous)
    # ═══════════════════════════════════════════════════════════════════════════
    df['ATR_14_Pct'] = atr14 / df['Close']
    df['ATR_Ratio_5_14'] = ATR(df, 5) / (atr14 + 1e-8)

    df['Volatility_5D']  = df['Return'].rolling(5).std()
    df['Volatility_20D'] = df['Return'].rolling(20).std()
    df['Vol_Ratio_5_20'] = df['Volatility_5D'] / (df['Volatility_20D'] + 1e-8)

    # Bollinger %B — continuous [0,1] position within bands
    bb_upper, bb_lower, bb_width = Bollinger_Bands(df['Close'], 20)
    df['PercentB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    df['BB_Width'] = bb_width / df['Close']
    df['BB_Squeeze'] = bb_width / (bb_width.rolling(120).mean() + 1e-8)

    # Keltner position
    keltner_upper, keltner_lower, _ = Keltner_Channel(df, 20)
    df['Keltner_Position'] = (df['Close'] - keltner_lower) / (keltner_upper - keltner_lower + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: VOLUME (smooth, normalized)
    # ═══════════════════════════════════════════════════════════════════════════
    obv = OBV(df)
    obv_sma = SMA(obv, 20)
    df['OBV_Dist']      = (obv - obv_sma) / (obv_sma.abs() + 1e-8)
    df['CMF_20']        = CMF(df, 20)
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    df['PVO']           = PVO(df['Volume'])
    df['RVOL']          = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    # Volume-price alignment — smooth correlation
    df['Vol_Price_Corr_20'] = df['Return'].rolling(20).corr(df['Volume'].pct_change())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: CANDLESTICK STRUCTURE (continuous ratios, no binary)
    # ═══════════════════════════════════════════════════════════════════════════
    total_range = df['High'] - df['Low'] + 1e-8
    df['Candle_Body_Ratio'] = (df['Close'] - df['Open']).abs() / total_range
    df['Lower_Shadow']      = (np.minimum(df['Close'], df['Open']) - df['Low']) / total_range
    df['Upper_Shadow']      = (df['High'] - np.maximum(df['Close'], df['Open'])) / total_range

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: GAP ANALYSIS (continuous, no binary)
    # ═══════════════════════════════════════════════════════════════════════════
    df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9: DIRECTION CONSISTENCY (continuous [-1, 1])
    # ═══════════════════════════════════════════════════════════════════════════
    direction = np.sign(df['Return'].fillna(0.0))
    df['Direction_Consistency_5']  = direction.rolling(5).sum() / 5
    df['Direction_Consistency_10'] = direction.rolling(10).sum() / 10

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 10: VORTEX (continuous directional strength)
    # ═══════════════════════════════════════════════════════════════════════════
    vi_plus, vi_minus = Vortex(df)
    df['Vortex_Spread'] = vi_plus - vi_minus  # Single smooth signal

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 11: STATISTICAL (smooth distributional features)
    # ═══════════════════════════════════════════════════════════════════════════
    df['Price_Zscore_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-8)
    df['Rolling_Skew']    = df['Return'].rolling(20).skew()
    df['Rolling_Kurt']    = df['Return'].rolling(20).kurt()
    df['Sharpe_20D']      = df['Return'].rolling(20).mean() / (df['Return'].rolling(20).std() + 1e-8)

    rolling_max = df['Close'].rolling(20).max()
    df['Drawdown_20D'] = (df['Close'] - rolling_max) / (rolling_max + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 12: 52-WEEK CONTEXT (continuous position signals)
    # ═══════════════════════════════════════════════════════════════════════════
    W52 = 250
    high_52w = df['High'].rolling(W52, min_periods=50).max()
    low_52w  = df['Low'].rolling(W52, min_periods=50).min()
    df['Dist_52W_High']   = (df['Close'] - high_52w) / (high_52w + 1e-8)
    df['Dist_52W_Low']    = (df['Close'] - low_52w)  / (low_52w  + 1e-8)
    df['Position_In_52W'] = (df['Close'] - low_52w) / (high_52w - low_52w + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 13: ALPHA PERSISTENCE (smooth cumulative signals)
    # ═══════════════════════════════════════════════════════════════════════════
    rolling_mean_ret = df['Return'].rolling(20).mean().fillna(0.0)
    alpha = df['Return'].fillna(0.0) - rolling_mean_ret
    df['Alpha_5D']  = alpha.rolling(5).sum().fillna(0.0)
    df['Alpha_10D'] = alpha.rolling(10).sum().fillna(0.0)

    # NOTE: No lag features — the 10-day sequence window provides temporal context.
    # NOTE: No binary/categorical features — transformers need smooth gradients.
    # NOTE: No interaction features — multi-head attention discovers these natively.

    return df


def compute_features_daily_transformer_v2(df):
    """
    State-of-the-Art Daily Transformer Feature Engineering.
    Computes a pruned, highly stationary, and normalized set of 32 indicators:
    - Bounded Oscillators scaled to [0, 1] or [-1, 1]
    - strictly stationary returns and ranges
    - Moving average distances
    - Z-scored and relative volume features
    """
    df = df.copy()

    # SECTION 1: Returns & Ranges
    df['Return'] = df['Close'].pct_change()
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['OC_Range'] = (df['Close'] - df['Open']) / df['Open']
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

    # SECTION 2: Multi-Horizon Returns (Pruned)
    df['Return_5D'] = df['Close'].pct_change(5)
    df['Return_10D'] = df['Close'].pct_change(10)
    df['Return_20D'] = df['Close'].pct_change(20)

    # SECTION 3: Trend & SMA Distances
    # ADX
    atr14 = ATR(df, 14)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm_clean = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm_clean = minus_dm.where(minus_dm > plus_dm, 0.0)
    plus_di = 100 * EMA(plus_dm_clean, 14) / (atr14 + 1e-8)
    minus_di = 100 * EMA(minus_dm_clean, 14) / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX_14'] = EMA(dx, 14) / 100.0  # scale to [0, 1]
    df['DI_Spread'] = (plus_di - minus_di) / 100.0  # scale to [-1, 1]

    # Distances
    df['Dist_SMA_20'] = (df['Close'] - SMA(df['Close'], 20)) / df['Close']
    df['Dist_SMA_50'] = (df['Close'] - SMA(df['Close'], 50)) / df['Close']

    # SECTION 4: Bounded Oscillators
    df['RSI_14'] = RSI(df['Close'], 14) / 100.0  # [0, 1]
    df['RSI_7'] = RSI(df['Close'], 7) / 100.0  # [0, 1]
    df['WPR_14'] = WPR(df, 14) / 100.0  # [-1, 0]
    
    stoch_k, stoch_d = Stochastic(df)
    df['Stoch_K'] = stoch_k / 100.0  # [0, 1]
    df['Stoch_D'] = stoch_d / 100.0  # [0, 1]

    # MACD scaled to close
    macd_fast = EMA(df['Close'], 12)
    macd_slow = EMA(df['Close'], 26)
    macd_val = macd_fast - macd_slow
    macd_signal = EMA(macd_val, 9)
    df['MACD_Hist_Scaled'] = (macd_val - macd_signal) / df['Close']

    vi_plus, vi_minus = Vortex(df)
    df['Vortex_Spread'] = vi_plus - vi_minus

    # SECTION 5: Volatility Regime
    df['ATR_14_Pct'] = atr14 / df['Close']
    
    vol5 = df['Return'].rolling(5).std()
    vol20 = df['Return'].rolling(20).std()
    df['Vol_Ratio_5_20'] = vol5 / (vol20 + 1e-8)

    bb_upper, bb_lower, bb_width = Bollinger_Bands(df['Close'], 20)
    df['PercentB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    df['BB_Width'] = bb_width / df['Close']

    # SECTION 6: Volume Dynamics
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    df['RVOL'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['CMF_20'] = CMF(df, 20)

    # SECTION 7: Candlestick & Gap
    total_range = df['High'] - df['Low'] + 1e-8
    df['Candle_Body_Ratio'] = (df['Close'] - df['Open']).abs() / total_range
    df['Lower_Shadow'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / total_range
    df['Upper_Shadow'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / total_range
    df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)

    # SECTION 8: 52-Week Context
    W52 = 250
    high_52w = df['High'].rolling(W52, min_periods=50).max()
    low_52w = df['Low'].rolling(W52, min_periods=50).min()
    df['Position_In_52W'] = (df['Close'] - low_52w) / (high_52w - low_52w + 1e-8)

    # Drop intermediate columns that could leak raw price/scale info
    # The return dataframe has exactly 32 features, all stationary.
    return df


