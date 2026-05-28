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
