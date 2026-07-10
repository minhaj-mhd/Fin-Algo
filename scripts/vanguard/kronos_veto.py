"""Kronos-base zero-shot veto layer for Vanguard (SHADOW-first).

⚠️ UNCERTIFIED — no Gauntlet run. Deployed for live observation per user decision
2026-07-03. Exploratory backtest evidence (see vault Conv-2026-07-02-Kronos-Zero-
Shot-Veto + data/research/kronos_veto/) is WEAK: post-cutoff keep-70% uplift
+1.73bps/trade (t≈2.0) on longs but inside the timing-null band; shorts ≈ nothing;
tighter filtering INVERTS. This layer must be treated as an experiment, not edge.

Rule (keep-70% operating point from the 2025-09-09→2026-06 OOS backtest window):
  LONG : keep iff p_up >= KRONOS_THR_LONG   (0.30)
  SHORT: keep iff 1-p_up >= KRONOS_THR_SHORT (0.4333)
where p_up = fraction of R=30 sampled Kronos-base forecast paths whose 4th
15m-bar close (= the 1h hold exit) is above the last completed bar's close.

Modes (config.KRONOS_VETO_MODE, env-overridable):
  shadow  -> score + log every candidate, NEVER blocks (default)
  enforce -> would_veto candidates are skipped by the orchestrator

Fail-safe: ANY error (model load, data, inference) => pass-through (no veto),
error logged. The layer can only ever *skip* trades, never add them.

Every decision is appended to data/kronos_veto_live.jsonl for offline
kept-vs-vetoed evaluation (the shadow counterfactual).
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.vanguard import config

KRONOS_DIR = r"C:\Users\loq\Desktop\Trading\Kronos"

_predictor = None
_load_failed = False


def _get_predictor():
    """Lazy singleton. Never raises; flips _load_failed so we never retry-loop."""
    global _predictor, _load_failed
    if _predictor is not None or _load_failed:
        return _predictor
    try:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        if KRONOS_DIR not in sys.path:
            sys.path.insert(0, KRONOS_DIR)
        import torch  # engine imports pandas long before any CUDA init
        from model import Kronos, KronosTokenizer, KronosPredictor
        t0 = time.time()
        tok = KronosTokenizer.from_pretrained(config.KRONOS_TOKENIZER_ID)
        mdl = Kronos.from_pretrained(config.KRONOS_MODEL_ID)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _predictor = KronosPredictor(mdl, tok, device=device, max_context=512)
        print(f"[KRONOS-VETO] loaded {config.KRONOS_MODEL_ID} on {device} in {time.time()-t0:.1f}s")
    except Exception as e:
        _load_failed = True
        print(f"[KRONOS-VETO] model load FAILED ({e!r}); layer inactive this session")
    return _predictor


def _normalize_candles(df):
    """Normalize the three live candle formats to [timestamp, open..volume].

    Sources: Upstox v3 REST / WS cache (lowercase + 'timestamp' column) and the
    yfinance fallback (DatetimeIndex, capitalized columns, sometimes MultiIndex).
    Also drops the trailing IN-PROGRESS 15m bar so the context matches the
    backtest spec (completed bars only). Raises ValueError when unusable.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "timestamp" not in df.columns:
        df = df.reset_index()
        for cand in ("Datetime", "Date", "index"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    df = df.rename(columns={c: c.lower() for c in df.columns})
    need = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"candles missing columns {missing} (got {list(df.columns)[:8]})")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # completed bars only: a bar labeled T covers T..T+15m
    cutoff = pd.Timestamp(datetime.now()) - pd.Timedelta(minutes=15)
    df = df[df["timestamp"] <= cutoff]
    return df[need]


def _log(rec):
    try:
        os.makedirs(os.path.dirname(config.KRONOS_VETO_LOG), exist_ok=True)
        with open(config.KRONOS_VETO_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # logging must never interfere with trading


def kronos_check(ticker, side, candles_df):
    """Score one trade candidate with Kronos-base.

    candles_df: completed 15m bars (timestamp, open, high, low, close, volume),
    oldest→newest, as returned by broker.get_recent_candles(interval='15minute').

    Returns dict(p_up, aligned, would_veto, mode, n_ctx, latency_s, error).
    would_veto is advisory in shadow mode; the orchestrator only acts on it in
    enforce mode. On any failure: would_veto=False (pass-through).
    """
    rec = dict(ts=datetime.now().isoformat(timespec="seconds"), ticker=ticker,
               side=side, p_up=None, aligned=None, would_veto=False,
               mode=config.KRONOS_VETO_MODE, n_ctx=0, latency_s=None, error=None)
    t0 = time.time()
    try:
        pred = _get_predictor()
        if pred is None:
            rec["error"] = "load_failed"
            return rec

        if candles_df is None or len(candles_df) == 0:
            rec["error"] = "no_candles"
            return rec
        df = _normalize_candles(candles_df)
        if len(df) < config.KRONOS_VETO_MIN_BARS:
            rec["error"] = f"insufficient_bars:{len(df)}"
            return rec
        df = df.tail(config.KRONOS_VETO_LOOKBACK).reset_index(drop=True)
        ts = df["timestamp"]
        x_df = df[["open", "high", "low", "close", "volume"]].astype(float)
        if x_df.isnull().values.any():
            rec["error"] = "nan_in_context"
            return rec
        rec["n_ctx"] = len(x_df)

        last_label = ts.iloc[-1]
        y_ts = pd.Series([last_label + pd.Timedelta(minutes=15 * (k + 1)) for k in range(4)])
        R = config.KRONOS_VETO_SAMPLES
        preds = pred.predict_batch([x_df] * R, [ts] * R, [y_ts] * R, pred_len=4,
                                   T=1.0, top_p=0.9, sample_count=1, verbose=False)
        last_close = float(x_df["close"].iloc[-1])
        rets = np.array([p["close"].iloc[-1] for p in preds], dtype=float) / last_close - 1.0

        p_up = float((rets > 0).mean())
        aligned = p_up if side == "LONG" else 1.0 - p_up
        thr = config.KRONOS_THR_LONG if side == "LONG" else config.KRONOS_THR_SHORT
        rec.update(p_up=round(p_up, 4), aligned=round(aligned, 4),
                   would_veto=bool(aligned < thr))
    except Exception as e:
        rec["error"] = repr(e)[:200]
        rec["would_veto"] = False  # fail-safe: never block on error
    finally:
        rec["latency_s"] = round(time.time() - t0, 2)
        _log(rec)
    return rec
