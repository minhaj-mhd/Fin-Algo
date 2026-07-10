"""
Daily OPTION-CHAIN OI forward-logger (works WITHOUT Upstox Plus).

Historical OI is paywalled (see memory: OI Plus paywall), but the live
/v2/option/chain snapshot is free. Run this daily (cron/manual, after close) to
accrue an OI history we can backtest later — the only no-Plus path to Gate 0.

Per F&O-eligible underlying it appends one row of aggregate OI features:
  spot, total_call_oi, total_put_oi, pcr_oi, max_pain, max_pain_dist,
  oi_conc_top3 (Herfindahl-ish), n_strikes, expiry

Output (append-only): data/oi_snapshots/option_chain_oi.csv
Run:    python scripts/collectors/log_option_chain_oi.py [--tickers RELIANCE,TCS] [--limit N]
Token:  UPSTOX_FULL_ACCESS_TOKEN (falls back to UPSTOX_ANALYTICS_ACCESS_TOKEN)
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
OUT = "data/oi_snapshots/option_chain_oi.csv"
CACHE = "scripts/instrument_cache.json"
RATE_PAUSE = 0.25


def _get(path, params):
    r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def nearest_expiry(eq_key):
    """Use /v2/option/contract to find the nearest non-expired expiry for an underlying."""
    code, body = _get("/v2/option/contract", {"instrument_key": eq_key})
    if body.get("status") != "success" or not body.get("data"):
        return None
    today = dt.date.today().isoformat()
    expiries = set()
    for c in body["data"]:
        e = c.get("expiry") if isinstance(c, dict) else c
        if e:
            expiries.add(e)
    future = sorted(e for e in expiries if e >= today)
    return future[0] if future else None


def aggregate_chain(rows, spot):
    strikes = sorted({r["strike_price"] for r in rows})
    call_oi = {r["strike_price"]: (r.get("call_options", {}).get("market_data", {}) or {}).get("oi", 0) or 0
               for r in rows}
    put_oi = {r["strike_price"]: (r.get("put_options", {}).get("market_data", {}) or {}).get("oi", 0) or 0
              for r in rows}
    tot_c = float(sum(call_oi.values()))
    tot_p = float(sum(put_oi.values()))
    # max pain: strike minimizing total option-writer payout at expiry == S
    def payout(S):
        return (sum(call_oi[k] * max(S - k, 0) for k in strikes) +
                sum(put_oi[k] * max(k - S, 0) for k in strikes))
    max_pain = min(strikes, key=payout) if strikes else float("nan")
    tot_oi = tot_c + tot_p
    by_strike = sorted(((call_oi[k] + put_oi[k]) for k in strikes), reverse=True)
    conc_top3 = float(sum(by_strike[:3]) / tot_oi) if tot_oi > 0 else float("nan")
    return {
        "spot": spot, "total_call_oi": tot_c, "total_put_oi": tot_p,
        "pcr_oi": (tot_p / tot_c) if tot_c > 0 else float("nan"),
        "max_pain": max_pain,
        "max_pain_dist": ((spot - max_pain) / spot) if spot else float("nan"),
        "oi_conc_top3": conc_top3, "n_strikes": len(strikes),
    }


def log_ticker(tkr, eq_key):
    exp = nearest_expiry(eq_key)
    if not exp:
        return None
    code, body = _get("/v2/option/chain", {"instrument_key": eq_key, "expiry_date": exp})
    data = body.get("data") if isinstance(body, dict) else None
    if not data:
        return None
    spot = data[0].get("underlying_spot_price")
    agg = aggregate_chain(data, spot)
    agg.update({"date": dt.date.today().isoformat(), "ticker": tkr, "expiry": exp})
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="comma list of base tickers; default = all in cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("FAIL: no Upstox token in .env")

    cache = json.load(open(CACHE))
    if args.tickers:
        names = [t.strip() for t in args.tickers.split(",")]
    else:
        names = [k for k, v in cache.items() if str(v).startswith("NSE_EQ")]
    if args.limit:
        names = names[:args.limit]

    out_rows, n_fno = [], 0
    for t in names:
        key = cache.get(t)
        if not key:
            continue
        try:
            row = log_ticker(t, key)
        except Exception as e:
            print(f"  {t}: error {e}")
            row = None
        if row:
            n_fno += 1
            out_rows.append(row)
            print(f"  {t}: spot={row['spot']} PCR={row['pcr_oi']:.2f} "
                  f"maxpain={row['max_pain']} ({row['max_pain_dist']:+.2%}) strikes={row['n_strikes']}")
        time.sleep(RATE_PAUSE)

    if not out_rows:
        print("No F&O chains returned (check token / market hours).")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(out_rows)
    cols = ["date", "ticker", "expiry", "spot", "total_call_oi", "total_put_oi",
            "pcr_oi", "max_pain", "max_pain_dist", "oi_conc_top3", "n_strikes"]
    df = df[cols]
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", header=header, index=False)
    print(f"\nappended {len(df)} rows ({n_fno} F&O names) -> {OUT}")


if __name__ == "__main__":
    main()
