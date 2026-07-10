"""
Feasibility probe for Upstox F&O Open-Interest collection (Gate-0 prerequisite).

Verifies — with artifacts, before building the full collector — two things:
  (a) ACCESS  : does UPSTOX_ANALYTICS_ACCESS_TOKEN have Upstox-Plus / expired-instruments entitlement?
  (b) DEPTH   : how far back does OI history actually go? (docs hint expiries are capped ~6 months)

It exercises both OI paths:
  Path 1 (futures candles) : /expiries -> /future/contract -> /historical-candle  (OI at field [6])
  Path 2 (option-chain OI) : /v2/market/oi?instrument_key&expiry&date            (by-date snapshot)

PRINT-ONLY. Writes nothing under data/, models/, or the vault.

Run:  python scripts/collectors/probe_upstox_oi.py
"""
import os
import sys
import json
import datetime as dt

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("UPSTOX_FULL_ACCESS_TOKEN") or os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
TOKEN_SRC = "UPSTOX_FULL_ACCESS_TOKEN" if os.getenv("UPSTOX_FULL_ACCESS_TOKEN") else "UPSTOX_ANALYTICS_ACCESS_TOKEN"
BASE = "https://api.upstox.com"
# RELIANCE underlying (from scripts/instrument_cache.json) — a deeply-liquid F&O name.
UNDERLYING = "NSE_EQ|INE002A01018"
UNDERLYING_NAME = "RELIANCE"

HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {TOKEN}"}


def _get(path, params=None):
    """GET helper that returns (ok, status_code, json_or_text)."""
    url = f"{BASE}{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = r.text[:500]
    return (r.status_code == 200 and isinstance(body, dict) and body.get("status") == "success",
            r.status_code, body)


def last_thursday(year, month):
    """NSE monthly F&O expiry (historically the last Thursday of the month)."""
    d = dt.date(year, month, 28)
    while d.month == month:
        d += dt.timedelta(days=1)
    d -= dt.timedelta(days=1)                       # last day of month
    while d.weekday() != 3:                          # 3 == Thursday
        d -= dt.timedelta(days=1)
    return d


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    if not TOKEN:
        print("FAIL: UPSTOX_ANALYTICS_ACCESS_TOKEN not set in .env"); sys.exit(1)
    print(f"Token loaded from {TOKEN_SRC} (len={len(TOKEN)}). Underlying: {UNDERLYING_NAME} = {UNDERLYING}")
    access_ok = futures_oi_ok = optchain_ok = False

    # ---- Probe A: expiry discovery + depth ---------------------------------
    banner("PROBE A  —  /expired-instruments/expiries  (access + depth)")
    ok, code, body = _get("/v2/expired-instruments/expiries", {"instrument_key": UNDERLYING})
    access_ok = ok
    expiries = []
    if ok:
        expiries = sorted(body.get("data", []))
        if expiries:
            earliest = dt.date.fromisoformat(expiries[0])
            months_back = (dt.date.today() - earliest).days / 30.4
            print(f"  PASS access. {len(expiries)} expiries returned.")
            print(f"  earliest={expiries[0]}  latest={expiries[-1]}  -> ~{months_back:.1f} months of depth")
        else:
            print("  PASS access but EMPTY expiry list.")
    else:
        print(f"  FAIL  http={code}  body={json.dumps(body)[:300]}")
        print("  (UDAPI100067 / 401 here => token lacks Upstox-Plus / expired-instruments entitlement.)")

    # ---- Probe B: future contract -> expired candle (OI presence) ----------
    banner("PROBE B  —  /future/contract  +  /historical-candle  (OI at field [6])")
    if expiries:
        test_expiry = expiries[len(expiries) // 2]   # a mid-range expired contract
        ok, code, body = _get("/v2/expired-instruments/future/contract",
                               {"instrument_key": UNDERLYING, "expiry_date": test_expiry})
        if ok and body.get("data"):
            con = body["data"][0]
            ikey = con["instrument_key"]
            print(f"  contract for {test_expiry}: {ikey}  (lot_size={con.get('lot_size')})")
            frm = (dt.date.fromisoformat(test_expiry) - dt.timedelta(days=60)).isoformat()
            ok2, code2, body2 = _get(
                f"/v2/expired-instruments/historical-candle/{ikey}/day/{test_expiry}/{frm}")
            if ok2:
                candles = body2.get("data", {}).get("candles", [])
                print(f"  candles returned: {len(candles)}  (range {frm} -> {test_expiry})")
                if candles:
                    c = candles[0]
                    print(f"  sample row (len={len(c)}): {c}")
                    has_oi = len(c) >= 7
                    print(f"  OI present at [6]? {'YES -> ' + str(c[6]) if has_oi else 'NO'}")
                    futures_oi_ok = has_oi
            else:
                print(f"  candle FAIL  http={code2}  body={json.dumps(body2)[:300]}")
        else:
            print(f"  future/contract FAIL  http={code}  body={json.dumps(body)[:300]}")
    else:
        print("  skipped (no expiries from Probe A)")

    # ---- Probe C: /v2/market/oi by historical date (deep-history claim) ----
    banner("PROBE C  —  /v2/market/oi  by historical date (option-chain OI depth)")
    target = dt.date.today() - dt.timedelta(days=300)     # ~10 months back
    exp = last_thursday(target.year, target.month)
    print(f"  testing date={target} with computed monthly expiry={exp}")
    ok, code, body = _get("/v2/market/oi",
                          {"instrument_key": UNDERLYING, "expiry": exp.isoformat(),
                           "date": target.isoformat()})
    d = body.get("data") if isinstance(body, dict) else None
    if ok and d:                                       # success AND non-null payload
        keys = list(d.keys()) if isinstance(d, dict) else type(d).__name__
        print(f"  PASS — option-chain OI served for a ~10-month-old date. data keys: {keys}")
        print(f"  raw (truncated): {json.dumps(d)[:400]}")
        optchain_ok = True
    elif ok:
        print(f"  INCONCLUSIVE — status=success but data={d!r} (likely wrong expiry/date or "
              f"endpoint also gated). NOT a pass.")
    else:
        print(f"  FAIL  http={code}  body={json.dumps(body)[:300]}")

    banner("VERDICT")
    print(f"  - expired-instruments ACCESS : {'PASS' if access_ok else 'FAIL (read-only token / no Plus)'}")
    print(f"  - futures-OI depth path      : {'PASS' if futures_oi_ok else 'BLOCKED (gated by access above)'}")
    print(f"  - option-chain-OI by-date    : {'PASS' if optchain_ok else 'NOT CONFIRMED'}")
    if not access_ok:
        print("  NEXT: supply a full-access (non read-only) daily access token with Upstox-Plus, then re-run.")


if __name__ == "__main__":
    main()
