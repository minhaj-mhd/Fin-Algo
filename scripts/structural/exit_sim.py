"""
Path-based exit simulator.

For each candidate (ticker, entry_ts, side, atr_at_entry, tp_mult, sl_mult):
  1. Walk 15m raw candles forward from entry_ts to the 1h boundary.
  2. Intrabar ATR TP/SL: if bar spans both -> SL-first (conservative).
  3. Conviction flip: exit at next open if 15m ranker loses support.
  4. 1h hard close: force-close at boundary bar close.

Returns DataFrame with exit_time, exit_price, gross_return, exit_reason.
"""
import numpy as np
import pandas as pd

EPS = 1e-8
COST_BPS = 10
COST = COST_BPS / 1e4

# Per-family TP/SL multiples and size factor
FAMILY_PARAMS = {
    101: dict(sl=1.5, tp=2.5, size=1.0),   # ORB
    102: dict(sl=1.2, tp=2.0, size=1.0),   # VWAP reclaim
    103: dict(sl=1.5, tp=3.0, size=1.0),   # RS burst
    104: dict(sl=1.5, tp=3.0, size=1.0),   # Squeeze breakout
    105: dict(sl=1.0, tp=1.5, size=0.5),   # Exhaustion reversal (smaller)
}


def _hour_boundary(ts):
    """Hold for up to 1 clock hour from entry (4-5 x 15m bars)."""
    return ts + pd.Timedelta(hours=1)


def simulate_trade(ticker_bars, entry_ts, side, atr_at_entry, strategy_id,
                   ml_scores=None, flip_long_pct=0.5, flip_short_rank=15):
    """
    Parameters
    ----------
    ticker_bars : DataFrame with columns ts, Open, High, Low, Close
                  for a single ticker, sorted ascending, tz-naive.
    entry_ts    : Timestamp (next bar open)
    side        : 'long' or 'short'
    atr_at_entry: float (ATR in price terms)
    strategy_id : int
    ml_scores   : DataFrame with ts, long_pct, short_rank (15m OOS per ticker/ts)
                  used for conviction-flip check. None => skip flip.
    flip_long_pct, flip_short_rank: flip thresholds

    Returns
    -------
    dict: entry_ts, exit_ts, entry_price, exit_price, gross_return, net_return,
          exit_reason, size_factor
    """
    fp = FAMILY_PARAMS.get(strategy_id, dict(sl=1.5, tp=2.0, size=1.0))
    sl_mult, tp_mult, size = fp['sl'], fp['tp'], fp['size']

    sgn = 1 if side == 'long' else -1
    boundary = _hour_boundary(entry_ts)

    # Slice forward bars from entry_ts
    hold_bars = ticker_bars[(ticker_bars['ts'] >= entry_ts) &
                            (ticker_bars['ts'] <= boundary)].copy()

    if len(hold_bars) == 0:
        return None

    entry_price = hold_bars.iloc[0]['Open']
    if entry_price <= 0:
        return None

    tp_price = entry_price + sgn * tp_mult * atr_at_entry
    sl_price = entry_price - sgn * sl_mult * atr_at_entry

    # Build ML score lookup if available
    ml_lookup = {}
    if ml_scores is not None and len(ml_scores) > 0:
        ml_lookup = dict(zip(ml_scores['ts'], zip(ml_scores['long_pct'], ml_scores['short_rank'])))

    exit_price = None
    exit_ts    = None
    exit_reason = None

    for i, row in hold_bars.iterrows():
        is_last = (row['ts'] == boundary) or (i == hold_bars.index[-1])

        # Check conviction flip at this bar's open (before TP/SL)
        if ml_scores is not None and row['ts'] in ml_lookup and not is_last:
            lp, sr = ml_lookup[row['ts']]
            if side == 'long' and (lp < flip_long_pct or sr <= flip_short_rank):
                exit_price  = row['Open']
                exit_ts     = row['ts']
                exit_reason = 'conviction_flip'
                break
            elif side == 'short' and lp > (1 - flip_long_pct):
                exit_price  = row['Open']
                exit_ts     = row['ts']
                exit_reason = 'conviction_flip'
                break

        # Intrabar TP/SL check
        if side == 'long':
            hit_sl = row['Low']  <= sl_price
            hit_tp = row['High'] >= tp_price
        else:
            hit_sl = row['High'] >= sl_price
            hit_tp = row['Low']  <= tp_price

        if hit_sl and hit_tp:
            # Both hit same bar -> SL-first (conservative)
            exit_price  = sl_price
            exit_ts     = row['ts']
            exit_reason = 'sl'
            break
        elif hit_sl:
            exit_price  = sl_price
            exit_ts     = row['ts']
            exit_reason = 'sl'
            break
        elif hit_tp:
            exit_price  = tp_price
            exit_ts     = row['ts']
            exit_reason = 'tp'
            break

        if is_last:
            exit_price  = row['Close']
            exit_ts     = row['ts']
            exit_reason = '1h_close'
            break

    if exit_price is None or exit_price <= 0:
        return None

    gross = sgn * (exit_price - entry_price) / entry_price
    net   = gross * size - COST

    return dict(
        entry_ts=entry_ts, exit_ts=exit_ts,
        entry_price=entry_price, exit_price=exit_price,
        gross_return=gross, net_return=net,
        exit_reason=exit_reason, size_factor=size,
    )


def run_simulation(triggers, raw_cache_dir, ml_scores_df=None):
    """
    Run exit simulation over all triggers.

    Parameters
    ----------
    triggers      : DataFrame from detectors (ts, Ticker, strategy_id, side, entry_ts)
    raw_cache_dir : path to per-ticker 15m OHLCV CSVs
    ml_scores_df  : optional DataFrame with (ts, Ticker, long_pct, short_rank)

    Returns
    -------
    DataFrame of completed trades.
    """
    import os, glob

    # Load all raw bars into a dict for fast lookup
    print("Loading raw 15m candles ...")
    all_bars = {}
    for ticker in triggers['Ticker'].unique():
        fname = os.path.join(raw_cache_dir, ticker.replace('.NS','') + '.csv')
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        df['ts'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df = df[['ts','Open','High','Low','Close']].drop_duplicates('ts').sort_values('ts')
        all_bars[ticker] = df

    # Build ML score lookup per ticker
    ml_by_ticker = {}
    if ml_scores_df is not None:
        for ticker, grp in ml_scores_df.groupby('Ticker'):
            ml_by_ticker[ticker] = grp[['ts','long_pct','short_rank']].sort_values('ts')

    print(f"Simulating {len(triggers):,} candidates ...")
    results = []
    for _, row in triggers.iterrows():
        ticker = row['Ticker']
        if ticker not in all_bars:
            continue

        bars = all_bars[ticker]
        ml   = ml_by_ticker.get(ticker, None)

        # Get ATR at the trigger bar
        trigger_bars = bars[bars['ts'] <= row['ts']]
        if len(trigger_bars) < 14:
            continue
        hl = trigger_bars['High'] - trigger_bars['Low']
        hc = (trigger_bars['High'] - trigger_bars['Close'].shift(1)).abs()
        lc = (trigger_bars['Low']  - trigger_bars['Close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if np.isnan(atr) or atr <= 0:
            continue

        trade = simulate_trade(
            bars, row['entry_ts'], row['side'], atr, row['strategy_id'],
            ml_scores=ml,
        )
        if trade is None:
            continue

        trade['trigger_ts']  = row['ts']
        trade['Ticker']      = ticker
        trade['strategy_id'] = row['strategy_id']
        trade['side']        = row['side']
        results.append(trade)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
