"""
Weekly market-cap rank for all TICKERS  (name + market cap + rank in a bracket)
================================================================================
Fetches each ticker's company name and market cap (yfinance, NSE .NS), ranks them
1 = largest, and PERSISTS a weekly snapshot. Designed to be run once a week.

Persistence:
  * data/marketcap_ranks.json                         -> latest snapshot (overwritten)
  * data/marketcap_history/marketcap_ranks_<ISOWEEK>.json  -> per-week archive

Each ticker is stored (and printed) with its rank in a bracket alongside the name:
  "RELIANCE.NS  Reliance Industries Limited (1)"

Run weekly:  python -m scripts.fetch_marketcap_ranks
"""
import os, sys, json, datetime as dt, warnings
import concurrent.futures as cf
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import yfinance as yf
from scripts.tickers import TICKERS

OUT_LATEST = "data/marketcap_ranks.json"
HIST_DIR   = "data/marketcap_history"
os.makedirs(HIST_DIR, exist_ok=True)


def fetch_one(ticker):
    """Return (ticker, company_name, market_cap|None). Robust to per-ticker failures."""
    name, mc = ticker.replace(".NS", ""), None
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        name = info.get("longName") or info.get("shortName") or name
        mc = info.get("marketCap")
        if mc is None:                                   # fallback to the lighter quote endpoint
            try:
                mc = tk.fast_info.get("market_cap") if hasattr(tk.fast_info, "get") else tk.fast_info["market_cap"]
            except Exception:
                mc = None
    except Exception:
        pass
    return ticker, name, (int(mc) if mc else None)


def main():
    print(f"Fetching market cap for {len(TICKERS)} tickers (yfinance)...")
    recs = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (tk, name, mc) in enumerate(ex.map(fetch_one, TICKERS), 1):
            recs[tk] = {"name": name, "market_cap": mc}
            if i % 25 == 0:
                print(f"  {i}/{len(TICKERS)}")

    # rank: 1 = largest market cap; tickers with no market cap go last, rank=None
    ranked = sorted((t for t in recs if recs[t]["market_cap"]),
                    key=lambda t: recs[t]["market_cap"], reverse=True)
    for r, tk in enumerate(ranked, 1):
        recs[tk]["rank"] = r
    missing = [t for t in recs if recs[t]["market_cap"] is None]
    for tk in missing:
        recs[tk]["rank"] = None

    today = dt.date.today()
    iso = today.isocalendar()               # (year, week, weekday)
    week_label = f"{iso[0]}-W{iso[1]:02d}"
    snapshot = {
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "week": week_label,
        "source": "yfinance",
        "count": len(recs),
        "ranked": len(ranked),
        "missing": missing,
        "tickers": recs,                    # ticker -> {name, market_cap, rank}
    }

    with open(OUT_LATEST, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    hist_path = os.path.join(HIST_DIR, f"marketcap_ranks_{week_label}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # ── readable list: rank in a bracket alongside the ticker name ──
    print(f"\n{'='*68}\n  MARKET-CAP RANK  |  {week_label}  |  {len(ranked)}/{len(recs)} ranked\n{'='*68}")
    for tk in ranked:
        r = recs[tk]["rank"]
        print(f"  {tk:<16} {recs[tk]['name']} ({r})")
    if missing:
        print("\n  [no market cap — ranked last]:")
        for tk in missing:
            print(f"  {tk:<16} {recs[tk]['name']} (—)")
    print(f"\n[SAVED] {OUT_LATEST}")
    print(f"[SAVED] {hist_path}")


if __name__ == "__main__":
    main()
