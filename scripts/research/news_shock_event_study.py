"""
News-shock proxy event study (EXPLORATORY - no verdict authority, not a Gauntlet run).

Question: if we "trade the news", how many tradeable events/day exist in our universe,
and does price DRIFT (news momentum) or FADE (mean reversion) after the shock, net of cost?

News proxy: a large IDIOSYNCRATIC 5-min return (stock return minus cross-sectional
median return that bar, so market-wide moves don't count) on abnormal volume
(rvol vs trailing 20-day median for the same time-of-day slot). A 1.5%+ idio move
in 5 minutes on 5x volume is, with near certainty, information arrival.

Data: data/raw_upstox_cache_5min_v3 (147 tickers, 2023-01..2026-06, left-labeled
bar starts 09:15..15:25, split-adjusted per v21 audit).

Hygiene:
- Intraday returns only (NaN across day boundary -> no overnight/gap contamination;
  the open gap is a separate, already-studied edge).
- Entries restricted to bar starts 09:30..14:30 (open window excluded - covered by
  gap-fade research; leaves >=1h of session to trade).
- De-clustered: first event per ticker per day for the P&L study.
- Entry variants: 'instant' (event-bar close) and 'delayed' (+1 bar = +5 min).
- Negative control: same ticker + same time slot on random non-event days, same
  sign-alignment -> exposes generic mean-reversion masquerading as a news effect.
- |ret|>15% dropped (circuit reopen / data error guard).

Outputs: data/research/news_shock/{cell_summary.csv, events_per_day.csv, report.txt}
"""
import os
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = "data/raw_upstox_cache_5min_v3"
OUT_DIR = "data/research/news_shock"
os.makedirs(OUT_DIR, exist_ok=True)

RET_THRESHOLDS = [0.010, 0.015, 0.020, 0.030]
RVOL_THRESHOLDS = [3.0, 5.0]
CANON_RET, CANON_RVOL = 0.015, 5.0
HORIZON_BARS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24}
COSTS_BPS = [6.0, 10.0]
MIN_CS_NAMES = 30
ENTRY_START, ENTRY_END = "09:30", "14:30"
MAX_ABS_RET = 0.15


def load_panel():
    frames = []
    for fp in sorted(glob.glob(os.path.join(SRC_DIR, "*.csv"))):
        tkr = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.drop(columns=["timestamp"]).drop_duplicates(subset="dt").sort_values("dt")
        df["ticker"] = tkr
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = panel["dt"].dt.date
    panel["slot"] = panel["dt"].dt.strftime("%H:%M")
    panel = panel[(panel["slot"] >= "09:15") & (panel["slot"] <= "15:25")]
    panel = panel.sort_values(["ticker", "dt"]).reset_index(drop=True)
    return panel


def add_features(panel):
    g = panel.groupby(["ticker", "date"], sort=False)
    panel["ret"] = g["close"].pct_change()  # NaN at first bar of each day -> no overnight
    panel.loc[panel["ret"].abs() > MAX_ABS_RET, "ret"] = np.nan

    # cross-sectional median return per bar (market move); require breadth
    cs = panel.groupby("dt")["ret"]
    panel["cs_med"] = cs.transform("median")
    panel["cs_n"] = cs.transform("count")
    panel["idio_ret"] = np.where(panel["cs_n"] >= MIN_CS_NAMES, panel["ret"] - panel["cs_med"], np.nan)

    # rvol vs trailing 20-day median volume for the same time slot (shifted 1 day)
    rvol_parts = []
    for tkr, tdf in panel.groupby("ticker", sort=False):
        pv = tdf.pivot_table(index="date", columns="slot", values="volume", aggfunc="first")
        base = pv.rolling(20, min_periods=10).median().shift(1)
        rv = (pv / base).stack().rename("rvol").reset_index()
        rv["ticker"] = tkr
        rvol_parts.append(rv)
    rvol = pd.concat(rvol_parts, ignore_index=True)
    panel = panel.merge(rvol, on=["ticker", "date", "slot"], how="left")

    # forward returns from event-bar close (instant) and from next-bar close (delayed +5min)
    panel = panel.sort_values(["ticker", "dt"]).reset_index(drop=True)
    g = panel.groupby(["ticker", "date"], sort=False)
    day_last = g["close"].transform("last")
    c1 = g["close"].shift(-1)
    for name, k in HORIZON_BARS.items():
        ck = g["close"].shift(-k).fillna(day_last)
        panel[f"fwd_{name}"] = ck / panel["close"] - 1
        ck_d = g["close"].shift(-(k + 1)).fillna(day_last)
        panel[f"fwdD_{name}"] = ck_d / c1 - 1
    panel["fwd_eod"] = day_last / panel["close"] - 1
    panel["fwdD_eod"] = day_last / c1 - 1
    panel["year"] = pd.to_datetime(panel["date"].astype(str)).dt.year
    return panel


def sign_aligned_stats(ev, col):
    x = (np.sign(ev["idio_ret"]) * ev[col]).dropna() * 1e4  # bps, drift(+)/fade(-)
    if len(x) < 20:
        return dict(n=len(x), mean=np.nan, t=np.nan, t_day=np.nan)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    daily = (np.sign(ev["idio_ret"]) * ev[col] * 1e4).groupby(ev["date"]).mean().dropna()
    t_day = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 10 else np.nan
    return dict(n=len(x), mean=x.mean(), t=t, t_day=t_day)


def main():
    print("loading 5-min cache...")
    panel = load_panel()
    n_days = panel["date"].nunique()
    print(f"panel: {len(panel):,} bars, {panel['ticker'].nunique()} tickers, {n_days} sessions")
    print("computing returns / idio / rvol / forward returns...")
    panel = add_features(panel)

    entry_ok = (panel["slot"] >= ENTRY_START) & (panel["slot"] <= ENTRY_END)
    eligible = panel["idio_ret"].notna() & panel["rvol"].notna()

    report, cell_rows, epd_rows = [], [], []
    for rt in RET_THRESHOLDS:
        for rv in RVOL_THRESHOLDS:
            is_ev = eligible & (panel["idio_ret"].abs() >= rt) & (panel["rvol"] >= rv)
            ev_all = panel[is_ev]
            ev = panel[is_ev & entry_ok].sort_values(["ticker", "dt"]).groupby(
                ["ticker", "date"], sort=False).head(1)  # de-clustered, tradeable window

            per_day = ev.groupby("date").size().reindex(
                sorted(panel["date"].unique()), fill_value=0)
            epd_rows.append(dict(ret_thr=rt, rvol_thr=rv,
                                 mean_per_day=per_day.mean(), median_per_day=per_day.median(),
                                 p90_per_day=per_day.quantile(0.90), pct_zero_days=(per_day == 0).mean(),
                                 total_events=len(ev), raw_bar_events=len(ev_all)))

            for entry, pfx in [("instant", "fwd"), ("delayed+5m", "fwdD")]:
                for hz in list(HORIZON_BARS) + ["eod"]:
                    s = sign_aligned_stats(ev, f"{pfx}_{hz}")
                    cell_rows.append(dict(ret_thr=rt, rvol_thr=rv, entry=entry, horizon=hz, **s))

    epd = pd.DataFrame(epd_rows)
    cells = pd.DataFrame(cell_rows)
    epd.to_csv(os.path.join(OUT_DIR, "events_per_day.csv"), index=False)
    cells.to_csv(os.path.join(OUT_DIR, "cell_summary.csv"), index=False)

    # ---------- report ----------
    report.append("=" * 78)
    report.append("NEWS-SHOCK PROXY EVENT STUDY (exploratory, no verdict authority)")
    report.append(f"universe: {panel['ticker'].nunique()} tickers | sessions: {n_days} "
                  f"({panel['date'].min()} .. {panel['date'].max()})")
    report.append("event = |idio 5-min ret| >= thr AND rvol >= thr, bar start 09:30-14:30,")
    report.append("        de-clustered to first event per ticker-day. Sign-aligned bps:")
    report.append("        POSITIVE mean = post-news DRIFT (momentum), NEGATIVE = FADE.")
    report.append("=" * 78)

    report.append("\n--- EVENTS PER DAY (answers 'how many news trades daily?') ---")
    report.append(f"{'ret_thr':>8} {'rvol':>5} {'mean/day':>9} {'med/day':>8} "
                  f"{'p90/day':>8} {'%zero':>6} {'total':>7}")
    for _, r in epd.iterrows():
        report.append(f"{r.ret_thr*100:>7.1f}% {r.rvol_thr:>5.0f} {r.mean_per_day:>9.2f} "
                      f"{r.median_per_day:>8.1f} {r.p90_per_day:>8.1f} "
                      f"{r.pct_zero_days*100:>5.1f}% {r.total_events:>7,.0f}")

    canon = (cells["ret_thr"] == CANON_RET) & (cells["rvol_thr"] == CANON_RVOL)
    report.append(f"\n--- CANONICAL CELL: |idio ret| >= {CANON_RET*100:.1f}%, rvol >= {CANON_RVOL:.0f} ---")
    report.append(f"{'entry':>12} {'horizon':>8} {'N':>7} {'gross bps':>10} {'t':>7} "
                  f"{'t(day)':>7} {'net@6':>7} {'net@10':>7}")
    for _, r in cells[canon].iterrows():
        report.append(f"{r.entry:>12} {r.horizon:>8} {r['n']:>7,.0f} {r['mean']:>10.2f} "
                      f"{r['t']:>7.2f} {r['t_day']:>7.2f} "
                      f"{r['mean']-COSTS_BPS[0]:>7.2f} {r['mean']-COSTS_BPS[1]:>7.2f}")

    report.append("\n--- ALL CELLS, 1h horizon (gross bps / t) ---")
    for entry in ["instant", "delayed+5m"]:
        report.append(f"  entry={entry}:")
        sub = cells[(cells["horizon"] == "1h") & (cells["entry"] == entry)]
        for _, r in sub.iterrows():
            report.append(f"    thr={r.ret_thr*100:.1f}% rvol>={r.rvol_thr:.0f}: "
                          f"N={r['n']:>6,.0f}  {r['mean']:>8.2f} bps  t={r['t']:>6.2f}")

    # direction / year / negative-control on the canonical cell
    is_ev = eligible & (panel["idio_ret"].abs() >= CANON_RET) & (panel["rvol"] >= CANON_RVOL)
    ev = panel[is_ev & entry_ok].sort_values(["ticker", "dt"]).groupby(
        ["ticker", "date"], sort=False).head(1)

    report.append("\n--- CANONICAL CELL SPLITS (instant entry, 1h / eod, gross bps) ---")
    for label, sub in [("pos-news (long drift test)", ev[ev["idio_ret"] > 0]),
                       ("neg-news (short drift test)", ev[ev["idio_ret"] < 0])]:
        s1, s2 = sign_aligned_stats(sub, "fwd_1h"), sign_aligned_stats(sub, "fwd_eod")
        report.append(f"  {label:<28} N={s1['n']:>6,}  1h: {s1['mean']:>7.2f} (t {s1['t']:.2f})"
                      f"  eod: {s2['mean']:>7.2f} (t {s2['t']:.2f})")
    for yr, sub in ev.groupby("year"):
        s1 = sign_aligned_stats(sub, "fwd_1h")
        report.append(f"  year {yr}:  N={s1['n']:>6,}  1h: {s1['mean']:>7.2f} bps (t {s1['t']:.2f})")

    # negative control: same ticker+slot, random non-event days, same sign-alignment
    rng = np.random.default_rng(42)
    non_ev = panel[eligible & entry_ok & ~is_ev & panel["idio_ret"].notna()]
    key_counts = ev.groupby(["ticker", "slot"]).size()
    pools = dict(tuple(non_ev.groupby(["ticker", "slot"])))
    ctrl_parts = []
    for key, cnt in key_counts.items():
        pool = pools.get(key)
        if pool is not None and len(pool):
            take = min(len(pool), cnt * 3)
            ctrl_parts.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])
    ctrl = pd.concat(ctrl_parts, ignore_index=True)
    report.append("\n--- NEGATIVE CONTROL (same ticker+slot, non-event bars, sign-aligned) ---")
    for hz in ["15m", "1h", "eod"]:
        se, sc = sign_aligned_stats(ev, f"fwd_{hz}"), sign_aligned_stats(ctrl, f"fwd_{hz}")
        report.append(f"  {hz:>4}: events {se['mean']:>7.2f} bps (t {se['t']:.2f}, N {se['n']:,})"
                      f"   control {sc['mean']:>7.2f} bps (t {sc['t']:.2f}, N {sc['n']:,})")

    txt = "\n".join(report)
    print(txt)
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nsaved -> {OUT_DIR}/report.txt, cell_summary.csv, events_per_day.csv")


if __name__ == "__main__":
    main()
