"""
Price-Action / Smart-Money-Concept (SMC) features — lookahead-free, OHLCV-only.

Implements the subset of the "hidden order-trigger" playbook that is COMPUTABLE
from the OHLCV bars this repo already owns (15m / 1h / daily Upstox 3y):

  * Candlestick patterns  — hammer, shooting star, bull/bear engulfing, doji,
    inside bar, morning/evening star + continuous wick/body geometry
    (the wick = where liquidity was grabbed).
  * Horizontal Support / Resistance — from CONFIRMED swing pivots: signed distance
    to nearest support/resistance, touch-count (level strength), round-number magnet.
  * Fair Value Gaps / 3-candle imbalances — formation flag + distance to nearest
    UNFILLED gap (the magnet that pulls price back).
  * Order blocks — last opposite candle before an impulsive displacement
    (institutional footprint), distance to nearest unmitigated block.
  * Liquidity sweeps / stop hunts — a prior swing high/low taken then reclaimed.
  * Displacement / impulse strength — body / ATR (size of the aggressive fill).

NOT here (needs data we do NOT have):
  * Volume-profile POC / VAH / VAL / LVN — needs intraday volume-at-price; the 5m
    3y cache is empty. (A crude approximation may be added later.)
  * Delta / footprint divergence — needs tick-level buy/sell volume — unavailable.
  * OI-magnet S/R — Upstox historical OI is paywalled (project_oi_plus_paywall).

CAUSALITY CONTRACT (enforced by the perturbation test in __main__):
  Every column at bar t uses ONLY bars <= t. Swing pivots are released with a
  `confirm`-bar lag (a fractal at index i becomes visible only at i+confirm).
  FVGs and order blocks are tracked forward-only. Corrupting any bar > t must
  never change a feature value at t.

Conventions match scripts/feature_utils.compute_features: input is a DataFrame
with Open/High/Low/Close/Volume (DateTime index); output is the SAME frame with
PA_* columns appended. All distances are normalised by price or ATR so they are
scale-free and survive the downstream per-query cross-sectional z-scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Public list of feature columns this module appends (import in the panel builder).
PA_FEATURES = [
    # — candlestick geometry / patterns —
    "PA_Body_Frac", "PA_Upper_Wick_Frac", "PA_Lower_Wick_Frac",
    "PA_Hammer", "PA_Shooting_Star", "PA_Bull_Engulf", "PA_Bear_Engulf",
    "PA_Doji", "PA_Inside_Bar", "PA_Morning_Star", "PA_Evening_Star",
    # — displacement / impulse —
    "PA_Displacement", "PA_Body_Z",
    # — support / resistance —
    "PA_Dist_Support", "PA_Dist_Resistance", "PA_Support_Touches",
    "PA_Res_Touches", "PA_Dist_Round",
    # — fair value gaps —
    "PA_FVG_Up", "PA_FVG_Down", "PA_Dist_Unfilled_FVG", "PA_In_FVG",
    # — order blocks —
    "PA_Dist_Bull_OB", "PA_Dist_Bear_OB", "PA_In_OB",
    # — liquidity sweeps / stop hunts —
    "PA_Sweep_High", "PA_Sweep_Low",
]


# ── basic geometry ──────────────────────────────────────────────────────────────
def _atr(high, low, close, window=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - pc).abs(),
                    (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _candlestick(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o)
    body_abs = body.abs()
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l

    body_frac = (body_abs / rng).fillna(0.0)
    upper_frac = (upper / rng).fillna(0.0)
    lower_frac = (lower / rng).fillna(0.0)

    bull = (c > o)
    bear = (c < o)

    # Hammer: long lower wick, small body near top, small upper wick.
    hammer = (lower_frac >= 0.5) & (body_frac <= 0.35) & (upper_frac <= 0.15)
    # Shooting star: long upper wick, small body near bottom.
    star = (upper_frac >= 0.5) & (body_frac <= 0.35) & (lower_frac <= 0.15)
    # Doji: body is a tiny fraction of range.
    doji = body_frac <= 0.1

    # Engulfing: current body fully engulfs prior body, opposite colour.
    po, pc = o.shift(1), c.shift(1)
    prev_bull = pc > po
    prev_bear = pc < po
    bull_engulf = bull & prev_bear & (c >= po) & (o <= pc)
    bear_engulf = bear & prev_bull & (o >= pc) & (c <= po)

    # Inside bar: high<prev_high and low>prev_low (compression before breakout).
    inside = (h < h.shift(1)) & (l > l.shift(1))

    # Morning / evening star (3-candle): big body, small middle, big opposite body.
    big = body_abs > body_abs.rolling(20, min_periods=5).mean()
    small_mid = (body_abs.shift(1) / rng.shift(1)).fillna(1.0) <= 0.4
    morning = (pc < po) & small_mid & bull & big & (c > (po.shift(1)))  # 3rd closes into 1st body
    evening = (pc > po) & small_mid & bear & big & (c < (po.shift(1)))

    out = pd.DataFrame(index=df.index)
    out["PA_Body_Frac"] = body_frac
    out["PA_Upper_Wick_Frac"] = upper_frac
    out["PA_Lower_Wick_Frac"] = lower_frac
    out["PA_Hammer"] = hammer.astype(np.float32)
    out["PA_Shooting_Star"] = star.astype(np.float32)
    out["PA_Bull_Engulf"] = bull_engulf.astype(np.float32)
    out["PA_Bear_Engulf"] = bear_engulf.astype(np.float32)
    out["PA_Doji"] = doji.astype(np.float32)
    out["PA_Inside_Bar"] = inside.astype(np.float32)
    out["PA_Morning_Star"] = morning.astype(np.float32)
    out["PA_Evening_Star"] = evening.astype(np.float32)
    return out


def _displacement(df, atr):
    body_abs = (df["Close"] - df["Open"]).abs()
    disp = (body_abs / (atr + 1e-9)).clip(upper=10.0)
    body_z = ((body_abs - body_abs.rolling(50, min_periods=10).mean())
              / (body_abs.rolling(50, min_periods=10).std() + 1e-9))
    out = pd.DataFrame(index=df.index)
    out["PA_Displacement"] = disp.fillna(0.0)
    out["PA_Body_Z"] = body_z.fillna(0.0)
    return out


# ── confirmed swing pivots (causal) ─────────────────────────────────────────────
def _confirmed_pivots(high, low, w=2):
    """Return per-bar arrays giving, for EACH bar t, the most recent confirmed swing
    high/low *level* and the index it was confirmed at (i+w). A fractal at i (high[i]
    is the max of [i-w, i+w]) is only visible at bar i+w — never before. Returns two
    lists of (confirm_idx, level) events sorted by confirm_idx."""
    H = high.to_numpy(); L = low.to_numpy()
    n = len(H)
    hi_events, lo_events = [], []
    for i in range(w, n - w):
        wnd_h = H[i - w:i + w + 1]
        wnd_l = L[i - w:i + w + 1]
        if H[i] == wnd_h.max() and (wnd_h.argmax() == w):
            hi_events.append((i + w, H[i]))   # visible only at i+w
        if L[i] == wnd_l.min() and (wnd_l.argmin() == w):
            lo_events.append((i + w, L[i]))
    return hi_events, lo_events


def _sr_features(df, atr, w=2, keep=12, touch_tol=0.3):
    """Distance to nearest support/resistance from confirmed pivots + touch-count."""
    close = df["Close"].to_numpy()
    atr_a = atr.to_numpy()
    n = len(close)
    hi_events, lo_events = _confirmed_pivots(df["High"], df["Low"], w=w)

    dist_sup = np.zeros(n, np.float32)
    dist_res = np.zeros(n, np.float32)
    sup_touch = np.zeros(n, np.float32)
    res_touch = np.zeros(n, np.float32)

    res_levels: list[float] = []   # confirmed swing highs (resistance candidates)
    sup_levels: list[float] = []   # confirmed swing lows (support candidates)
    jh = jl = 0
    for t in range(n):
        while jh < len(hi_events) and hi_events[jh][0] <= t:
            res_levels.append(hi_events[jh][1]); jh += 1
            if len(res_levels) > keep:
                res_levels.pop(0)
        while jl < len(lo_events) and lo_events[jl][0] <= t:
            sup_levels.append(lo_events[jl][1]); jl += 1
            if len(sup_levels) > keep:
                sup_levels.pop(0)
        c = close[t]; a = atr_a[t] if atr_a[t] > 0 else (abs(c) * 1e-3 + 1e-9)
        # nearest resistance ABOVE, support BELOW (signed, ATR-normalised, magnitude)
        above = [r for r in res_levels if r >= c]
        below = [s for s in sup_levels if s <= c]
        if above:
            r = min(above); dist_res[t] = (r - c) / a
            res_touch[t] = sum(1 for x in res_levels if abs(x - r) <= touch_tol * a)
        else:
            dist_res[t] = 5.0
        if below:
            s = max(below); dist_sup[t] = (c - s) / a
            sup_touch[t] = sum(1 for x in sup_levels if abs(x - s) <= touch_tol * a)
        else:
            dist_sup[t] = 5.0

    out = pd.DataFrame(index=df.index)
    out["PA_Dist_Support"] = np.clip(dist_sup, 0, 10)
    out["PA_Dist_Resistance"] = np.clip(dist_res, 0, 10)
    out["PA_Support_Touches"] = sup_touch
    out["PA_Res_Touches"] = res_touch
    # round-number magnet: distance to nearest 1% grid level, ATR-normalised
    step = np.where(close >= 1000, 50.0, np.where(close >= 100, 5.0, 0.5))
    nearest = np.round(close / step) * step
    out["PA_Dist_Round"] = np.clip(np.abs(close - nearest) / (atr_a + 1e-9), 0, 10)
    return out


# ── fair value gaps (causal, forward-only tracking) ─────────────────────────────
def _fvg_features(df, atr, max_track=20):
    h = df["High"].to_numpy(); l = df["Low"].to_numpy(); c = df["Close"].to_numpy()
    atr_a = atr.to_numpy(); n = len(c)
    fvg_up = np.zeros(n, np.float32); fvg_dn = np.zeros(n, np.float32)
    dist_unfilled = np.full(n, 5.0, np.float32); in_fvg = np.zeros(n, np.float32)

    gaps: list[tuple[float, float]] = []   # (low_edge, high_edge) of unfilled gaps
    for t in range(n):
        # mitigate existing gaps with the CURRENT bar's range (known at t)
        if gaps:
            gaps = [(lo, hi) for (lo, hi) in gaps if not (l[t] <= hi and h[t] >= lo)]
        # detect a gap formed by bars t-2,t-1,t (all <= t, causal)
        if t >= 2:
            if l[t] > h[t - 2]:                       # bullish FVG: gap between t-2 high and t low
                fvg_up[t] = 1.0; gaps.append((h[t - 2], l[t]))
            elif h[t] < l[t - 2]:                     # bearish FVG
                fvg_dn[t] = 1.0; gaps.append((h[t], l[t - 2]))
            if len(gaps) > max_track:
                gaps = gaps[-max_track:]
        # distance to nearest unfilled gap midpoint; flag if price sits inside one
        a = atr_a[t] if atr_a[t] > 0 else (abs(c[t]) * 1e-3 + 1e-9)
        if gaps:
            mids = np.array([(lo + hi) / 2 for (lo, hi) in gaps])
            dist_unfilled[t] = min(np.min(np.abs(mids - c[t]) / a), 10.0)
            in_fvg[t] = 1.0 if any(lo <= c[t] <= hi for (lo, hi) in gaps) else 0.0

    out = pd.DataFrame(index=df.index)
    out["PA_FVG_Up"] = fvg_up
    out["PA_FVG_Down"] = fvg_dn
    out["PA_Dist_Unfilled_FVG"] = dist_unfilled
    out["PA_In_FVG"] = in_fvg
    return out


# ── order blocks (causal) ───────────────────────────────────────────────────────
def _order_block_features(df, atr, disp_mult=1.5, max_track=10):
    o = df["Open"].to_numpy(); c = df["Close"].to_numpy()
    h = df["High"].to_numpy(); l = df["Low"].to_numpy()
    atr_a = atr.to_numpy(); n = len(c)
    dist_bull = np.full(n, 5.0, np.float32); dist_bear = np.full(n, 5.0, np.float32)
    in_ob = np.zeros(n, np.float32)
    body = np.abs(c - o)

    bull_obs: list[tuple[float, float]] = []   # (low, high) price zone of bullish OBs
    bear_obs: list[tuple[float, float]] = []
    for t in range(n):
        # an impulsive UP bar at t whose prior bar t-1 was DOWN -> bar t-1 is a bullish OB zone
        if t >= 1 and body[t] >= disp_mult * (atr_a[t] + 1e-9):
            if c[t] > o[t] and c[t - 1] < o[t - 1]:
                bull_obs.append((l[t - 1], h[t - 1]))
            elif c[t] < o[t] and c[t - 1] > o[t - 1]:
                bear_obs.append((l[t - 1], h[t - 1]))
            bull_obs = bull_obs[-max_track:]; bear_obs = bear_obs[-max_track:]
        # mitigate (remove) blocks the current bar has traded back into
        bull_obs = [(lo, hi) for (lo, hi) in bull_obs if not (l[t] <= hi and h[t] >= lo)] \
            if False else bull_obs   # keep zones; "in_ob" flags retest instead of deleting
        a = atr_a[t] if atr_a[t] > 0 else (abs(c[t]) * 1e-3 + 1e-9)
        if bull_obs:
            mids = np.array([(lo + hi) / 2 for (lo, hi) in bull_obs])
            dist_bull[t] = min(np.min(np.abs(mids - c[t]) / a), 10.0)
        if bear_obs:
            mids = np.array([(lo + hi) / 2 for (lo, hi) in bear_obs])
            dist_bear[t] = min(np.min(np.abs(mids - c[t]) / a), 10.0)
        inside = any(lo <= c[t] <= hi for (lo, hi) in bull_obs) or \
                 any(lo <= c[t] <= hi for (lo, hi) in bear_obs)
        in_ob[t] = 1.0 if inside else 0.0

    out = pd.DataFrame(index=df.index)
    out["PA_Dist_Bull_OB"] = dist_bull
    out["PA_Dist_Bear_OB"] = dist_bear
    out["PA_In_OB"] = in_ob
    return out


# ── liquidity sweeps / stop hunts (causal) ──────────────────────────────────────
def _sweep_features(df, lookback=10):
    h = df["High"]; l = df["Low"]; c = df["Close"]
    prior_high = h.shift(1).rolling(lookback, min_periods=2).max()
    prior_low = l.shift(1).rolling(lookback, min_periods=2).min()
    # sweep high: bar pokes above prior swing high but closes back below it (rejection)
    sweep_high = (h > prior_high) & (c < prior_high)
    sweep_low = (l < prior_low) & (c > prior_low)
    out = pd.DataFrame(index=df.index)
    out["PA_Sweep_High"] = sweep_high.fillna(False).astype(np.float32)
    out["PA_Sweep_Low"] = sweep_low.fillna(False).astype(np.float32)
    return out


# ── public entry point ──────────────────────────────────────────────────────────
def add_price_action_features(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Append all PA_* columns to a frame holding Open/High/Low/Close/Volume.
    Lookahead-free; safe to call on the same per-ticker frame fed to compute_features."""
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        raise ValueError(f"price_action needs {need}, got {set(df.columns)}")
    atr = _atr(df["High"], df["Low"], df["Close"], atr_window)
    parts = [
        _candlestick(df),
        _displacement(df, atr),
        _sr_features(df, atr),
        _fvg_features(df, atr),
        _order_block_features(df, atr),
        _sweep_features(df),
    ]
    pa = pd.concat(parts, axis=1)
    # numeric hygiene; downstream z-scoring handles scale
    pa = pa.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    assert list(pa.columns) == PA_FEATURES, "PA column/order mismatch"
    return pd.concat([df, pa], axis=1)


# ── smoke test + causality (perturbation) verification ──────────────────────────
def _smoke_test():
    rng = np.random.default_rng(0)
    n = 600
    ret = rng.normal(0, 0.01, n)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame({"Open": open_, "High": np.maximum.reduce([high, open_, close]),
                       "Low": np.minimum.reduce([low, open_, close]), "Close": close,
                       "Volume": rng.integers(1e4, 1e6, n)}, index=idx)

    out = add_price_action_features(df)
    assert all(col in out.columns for col in PA_FEATURES)
    assert np.isfinite(out[PA_FEATURES].to_numpy()).all(), "non-finite PA feature"
    print(f"[smoke] {len(PA_FEATURES)} PA features, all finite. shape={out.shape}")
    print(out[PA_FEATURES].describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    # CAUSALITY: corrupt the tail (bars > t0); features at <= t0 must be identical.
    t0 = 400
    df2 = df.astype(float).copy()
    df2.iloc[t0 + 1:] = df2.iloc[t0 + 1:] * 1.5   # arbitrary future corruption
    out2 = add_price_action_features(df2)
    a = out[PA_FEATURES].iloc[:t0 + 1].to_numpy()
    b = out2[PA_FEATURES].iloc[:t0 + 1].to_numpy()
    max_diff = np.nanmax(np.abs(a - b))
    assert max_diff < 1e-6, f"LOOKAHEAD LEAK: features changed by {max_diff} when future was altered"
    print(f"[causality] OK — corrupting bars > {t0} left all features <= {t0} unchanged "
          f"(max diff {max_diff:.2e})")


if __name__ == "__main__":
    _smoke_test()
