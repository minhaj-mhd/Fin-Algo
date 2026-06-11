"""
Pure deterministic detectors for the 5 structural families.

All detectors:
  - operate on completed candles only
  - entry is NEXT bar open (entry_time = trigger bar + 1 bar)
  - return DataFrame of triggers with columns:
      ts, Ticker, strategy_id, side, entry_ts

Family IDs:
  101 = ORB expansion
  102 = VWAP reclaim / reject
  103 = Relative-strength burst
  104 = Liquidity-squeeze breakout
  105 = Exhaustion reversal
"""
import pandas as pd
import numpy as np

EPS = 1e-8
NEXT_BAR_OFFSET = pd.Timedelta(minutes=15)


def _next_bar(ts_series):
    return ts_series + NEXT_BAR_OFFSET


def _trigger_df(panel_subset, strategy_id, side):
    df = panel_subset[['ts','Ticker']].copy()
    df['strategy_id'] = strategy_id
    df['side'] = side
    df['entry_ts'] = _next_bar(df['ts'])
    return df.reset_index(drop=True)


# ── 101: ORB Expansion ────────────────────────────────────────────────────────

def orb_expansion(panel, side='long'):
    """Fire after 10:15 IST when price breaks the 09:15-09:30 opening range."""
    p = panel.copy()
    # Only consider bars at/after 10:15 (after the opening range is fully formed)
    valid_time = p['tod'] >= '10:15'

    if side == 'long':
        cond = (
            valid_time &
            p['or_high'].notna() &
            (p['Close'] > p['or_high']) &
            (p['Volume'] > 1.5 * p['avg_vol_20']) &
            (p['ret_30m'] > 0) &
            (p['relative_ret_60m'] > 0) &
            (p['Close'] > p['session_vwap'])
        )
    else:
        cond = (
            valid_time &
            p['or_low'].notna() &
            (p['Close'] < p['or_low']) &
            (p['Volume'] > 1.5 * p['avg_vol_20']) &
            (p['ret_30m'] < 0) &
            (p['relative_ret_60m'] < 0) &
            (p['Close'] < p['session_vwap'])
        )

    return _trigger_df(p[cond], 101, side)


# ── 102: VWAP Reclaim / Reject ────────────────────────────────────────────────

def vwap_reclaim(panel, side='long'):
    """Price crosses VWAP with force: previous bar on wrong side, current reclaims."""
    p = panel.copy()

    if side == 'long':
        cond = (
            p['prev_close'].notna() &
            (p['prev_close'] < p['prev_vwap']) &                    # was below
            (p['Close'] > p['session_vwap']) &                      # now above
            (p['Close'] > p['Open']) &                              # green candle
            (p['Low'] <= p['session_vwap'] * 1.002) &               # wicked through
            (p['rvol_20'] >= 1.2) &
            (p['relative_ret_30m'] > 0)
        )
    else:
        cond = (
            p['prev_close'].notna() &
            (p['prev_close'] > p['prev_vwap']) &
            (p['Close'] < p['session_vwap']) &
            (p['Close'] < p['Open']) &
            (p['High'] >= p['session_vwap'] * 0.998) &
            (p['rvol_20'] >= 1.2) &
            (p['relative_ret_30m'] < 0)
        )

    return _trigger_df(p[cond], 102, side)


# ── 103: Relative-Strength Burst ─────────────────────────────────────────────

def rs_burst(panel, side='long'):
    """Stock outperforms market meaningfully over the last hour with volume confirmation."""
    p = panel.copy()

    if side == 'long':
        cond = (
            (p['ret_60m'] > 0.004) &
            (p['relative_ret_60m'] > 0.0025) &
            (p['rvol_20'] >= 1.5) &
            (p['Close'] > p['session_vwap']) &
            (p['Close'] > p['roll_high_4']) &
            (p['nifty_ret_60m'] >= -0.0015)
        )
    else:
        cond = (
            (p['ret_60m'] < -0.004) &
            (p['relative_ret_60m'] < -0.0025) &
            (p['rvol_20'] >= 1.5) &
            (p['Close'] < p['session_vwap']) &
            (p['Close'] < p['roll_low_4']) &
            (p['nifty_ret_60m'] <= 0.0015)
        )

    return _trigger_df(p[cond], 103, side)


# ── 104: Liquidity-Squeeze Breakout ──────────────────────────────────────────

def squeeze_breakout(panel, side='long'):
    """Compression into expansion: low ATR period followed by a high-volume range expansion."""
    p = panel.copy()

    if side == 'long':
        cond = (
            p['atr_pct_4bar'].notna() &
            p['atr_pct_20bar'].notna() &
            (p['atr_pct_4bar'] < 0.75 * p['atr_pct_20bar']) &
            (p['range_pct'] > 1.3 * p['avg_range_pct_20']) &
            (p['Close'] > p['roll_high_6']) &
            (p['Volume'] > 1.8 * p['avg_vol_20']) &
            (p['close_location'] >= 0.70)
        )
    else:
        cond = (
            p['atr_pct_4bar'].notna() &
            p['atr_pct_20bar'].notna() &
            (p['atr_pct_4bar'] < 0.75 * p['atr_pct_20bar']) &
            (p['range_pct'] > 1.3 * p['avg_range_pct_20']) &
            (p['Close'] < p['roll_low_6']) &
            (p['Volume'] > 1.8 * p['avg_vol_20']) &
            (p['close_location'] <= 0.30)
        )

    return _trigger_df(p[cond], 104, side)


# ── 105: Exhaustion Reversal ──────────────────────────────────────────────────

def exhaustion_reversal(panel, side='long'):
    """Mean-reversion: sharp move with a long rejection wick that reclaims most of the range."""
    p = panel.copy()
    prev_low_15m  = p['Low'].shift(1)
    prev_high_15m = p['High'].shift(1)

    if side == 'long':
        support_reclaim = (
            (p['Close'] >= prev_low_15m * (1 - 0.004)) &
            p['Close'].notna()
        )
        cond = (
            (p['ret_30m'] < -0.006) &
            (p['lower_wick_pct'] >= 0.45) &
            (p['close_location'] >= 0.65) &
            (p['rvol_20'] >= 1.4) &
            (p['Close'] >= p['Low'] + 0.6 * (p['High'] - p['Low'])) &
            support_reclaim
        )
    else:
        resist_reject = (
            (p['Close'] <= prev_high_15m * (1 + 0.004)) &
            p['Close'].notna()
        )
        cond = (
            (p['ret_30m'] > 0.006) &
            (p['upper_wick_pct'] >= 0.45) &
            (p['close_location'] <= 0.35) &
            (p['rvol_20'] >= 1.4) &
            resist_reject
        )

    return _trigger_df(p[cond], 105, side)


# ── Composite trigger (all families) ─────────────────────────────────────────

DETECTORS = {
    101: orb_expansion,
    102: vwap_reclaim,
    103: rs_burst,
    104: squeeze_breakout,
    105: exhaustion_reversal,
}

FAMILY_NAMES = {
    101: 'ORB Expansion',
    102: 'VWAP Reclaim',
    103: 'RS Burst',
    104: 'Squeeze Breakout',
    105: 'Exhaustion Reversal',
}


def run_all_detectors(panel, sides=('long',)):
    """Run all families and return combined trigger DataFrame."""
    frames = []
    for sid, fn in DETECTORS.items():
        for side in sides:
            triggers = fn(panel, side=side)
            if len(triggers) > 0:
                frames.append(triggers)
    if not frames:
        return pd.DataFrame(columns=['ts','Ticker','strategy_id','side','entry_ts'])
    return pd.concat(frames, ignore_index=True)
