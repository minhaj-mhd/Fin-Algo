"""
Historical FUTURES Open-Interest collector (REQUIRES Upstox Plus).

This is the cheap deep-history OI path: per underlying, one historical-candle call
per monthly contract returns that contract's full daily OI series (OI at field [6]).
We stitch consecutive front-month futures into a continuous daily OI + basis series.

  /v2/expired-instruments/expiries        -> past expiry dates
  /v2/expired-instruments/future/contract -> expired futures instrument_key
  /v2/expired-instruments/historical-candle/{key}/day/{to}/{from} -> [ts,o,h,l,c,vol,OI]

STATUS: blocked until the account has Upstox Plus (else UDAPI1149 / UDAPI100067).
Fails gracefully with a clear message — ready to run the moment Plus is active.
Output: data/raw_upstox_oi_cache/<TICKER>_futures_oi.csv  (per-ticker cache)
Run:    python scripts/collectors/collect_upstox_oi.py [--tickers RELIANCE,TCS] [--limit N]
Token:  UPSTOX_FULL_ACCESS_TOKEN (full-access, non-extended; see upstox_login.py)
"""
import os
import sys
import json
import time
import argparse
import datetime as dt

import requests
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.getcwd())
load_dotenv(os.path.join(os.getcwd(), ".env"))

BASE = "https://api.upstox.com"
TOKEN = os.getenv("UPSTOX_FULL_ACCESS_TOKEN") or os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {TOKEN}"}
CACHE = "scripts/instrument_cache.json"
OUT_DIR = "data/raw_upstox_oi_cache"
RATE_PAUSE = 0.25


class PlusRequired(Exception):
    pass


def _get(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = {}
    if r.status_code in (401, 403):
        errs = body.get("errors", [{}])
        code = errs[0].get("errorCode", "")
        if code in ("UDAPI1149", "UDAPI100067"):
            raise PlusRequired(errs[0].get("message", "Plus/full-access required"))
    return r.status_code, body


def get_expiries(underlying):
    _, body = _get("/v2/expired-instruments/expiries", {"instrument_key": underlying})
    return sorted(body.get("data", [])) if body.get("status") == "success" else []


def get_future_key(underlying, expiry):
    _, body = _get("/v2/expired-instruments/future/contract",
                   {"instrument_key": underlying, "expiry_date": expiry})
    data = body.get("data") or []
    return data[0]["instrument_key"] if data else None


def get_candles(inst_key, frm, to):
    safe = requests.utils.quote(inst_key, safe="")
    _, body = _get(f"/v2/expired-instruments/historical-candle/{safe}/day/{to}/{frm}")
    candles = (body.get("data") or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["ts", "o", "h", "l", "c", "vol", "oi"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    return df[["ts", "c", "vol", "oi"]].sort_values("ts")


def collect_underlying(tkr, underlying):
    """Stitch front-month futures into a continuous daily OI series."""
    expiries = get_expiries(underlying)
    if not expiries:
        return pd.DataFrame()
    frames, prev = [], None
    for exp in expiries:
        key = get_future_key(underlying, exp)
        time.sleep(RATE_PAUSE)
        if not key:
            continue
        # front-month window: (prev_expiry, this_expiry]
        frm = (dt.date.fromisoformat(prev) + dt.timedelta(days=1)).isoformat() if prev else \
            (dt.date.fromisoformat(exp) - dt.timedelta(days=40)).isoformat()
        df = get_candles(key, frm, exp)
        time.sleep(RATE_PAUSE)
        if not df.empty:
            df["expiry"] = exp
            frames.append(df)
        prev = exp
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).drop_duplicates("ts", keep="last").sort_values("ts")
    out = out.rename(columns={"c": "fut_close", "vol": "fut_vol", "oi": "fut_oi"})
    out["ticker"] = tkr
    out["fut_oi_chg"] = out["fut_oi"].diff()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("FAIL: no Upstox token in .env")

    cache = json.load(open(CACHE))
    names = ([t.strip() for t in args.tickers.split(",")] if args.tickers
             else [k for k, v in cache.items() if str(v).startswith("NSE_EQ")])
    if args.limit:
        names = names[:args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)

    done = 0
    for t in names:
        key = cache.get(t)
        if not key:
            continue
        try:
            df = collect_underlying(t, key)
        except PlusRequired as e:
            print(f"\nBLOCKED: {e}\n  -> expired-instruments OI requires Upstox Plus. "
                  f"Buy Plus, re-mint token (upstox_login.py), then re-run. Nothing written.")
            return
        if df.empty:
            print(f"  {t}: no futures OI returned")
            continue
        path = os.path.join(OUT_DIR, f"{t}_futures_oi.csv")
        df.to_csv(path, index=False)
        done += 1
        print(f"  {t}: {len(df)} days  OI range {df['ts'].min().date()}..{df['ts'].max().date()} -> {path}")
        time.sleep(RATE_PAUSE)
    print(f"\ncollected {done}/{len(names)} tickers")


if __name__ == "__main__":
    main()
