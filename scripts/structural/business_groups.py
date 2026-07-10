"""
Business-group (promoter-house) mapping for the 172-ticker NSE universe.

This is the EXOGENOUS edge source for the structural graph (Gate 1). Unlike
price-correlation / lead-lag edges (which the CST work showed are arbitraged away
at 1h), business-group co-membership is a slow, structural prior NOT derivable from
short-horizon price action — Adani/Tata/etc. names share sponsor-level risk, group
news, and cross-holdings that drive co-movement beyond technical correlation.

Only multi-node groups (>=2 tickers in our universe) are listed — a single-node
group contributes no edge. Keyed on BASE ticker (no .NS). Curated from public
promoter-group knowledge; treat as reference data, not a verified filing extract.
"""

# group label -> base tickers in our universe
BUSINESS_GROUPS = {
    "TATA":      ["TCS", "TATASTEEL", "TATAPOWER", "TATACONSUM", "TATACOMM",
                  "TATACHEM", "TITAN", "TRENT", "VOLTAS"],
    "ADANI":     ["ADANIENT", "ADANIPORTS", "ADANIPOWER", "ADANIGREEN",
                  "ATGL", "AWL", "AMBUJACEM"],
    "BAJAJ":     ["BAJFINANCE", "BAJAJFINSV", "BAJAJ-AUTO"],
    "ICICI":     ["ICICIBANK", "ICICIGI", "ICICIPRULI"],
    "SBI":       ["SBIN", "SBICARD", "SBILIFE"],
    "HDFC":      ["HDFCBANK", "HDFCLIFE"],
    "MAHINDRA":  ["M&M", "M&MFIN", "TECHM"],        # incl. Tech Mahindra
    "BIRLA":     ["GRASIM", "ULTRACEMCO", "HINDALCO", "ABCAPITAL"],
    "GODREJ":    ["GODREJCP", "GODREJPROP"],
    "MURUGAPPA": ["COROMANDEL", "CHOLAFIN"],
    "LT":        ["LT", "LTTS"],
}

# Deliberately kept SEPARATE (common-name collisions / split houses), documented
# so a future editor doesn't "helpfully" merge them:
#   JSWSTEEL (Sajjan Jindal) != JINDALSTEL (Naveen Jindal) — different groups
#   APOLLOHOSP (Apollo Hospitals) != APOLLOTYRE (Apollo Tyres) — unrelated

# base ticker -> group label
TICKER_TO_GROUP = {t: g for g, ts in BUSINESS_GROUPS.items() for t in ts}


def base(ticker: str) -> str:
    """Normalize a ticker to its base name (strip .NS / exchange suffix)."""
    return ticker.replace(".NS", "").strip()


def group_of(ticker: str):
    """Return the business-group label for a ticker, or None if ungrouped."""
    return TICKER_TO_GROUP.get(base(ticker))


if __name__ == "__main__":
    n_groups = len(BUSINESS_GROUPS)
    n_members = sum(len(v) for v in BUSINESS_GROUPS.values())
    print(f"{n_groups} multi-node business groups covering {n_members} tickers")
    for g, ts in BUSINESS_GROUPS.items():
        print(f"  {g:10s} ({len(ts)}): {', '.join(ts)}")
