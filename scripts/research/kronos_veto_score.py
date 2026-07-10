"""Zero-shot Kronos-base veto scores for the 1h ranker book (dualtf trade panel).

Research question: do zero-shot next-hour forecasts from Kronos-base
(NeoQuasar/Kronos-base, 102.3M, 12B-bar / 45-exchange pretrained prior) carry
veto-grade information on the 1h book's top-K picks that our from-scratch veto
models (co-sign v10, daily veto, BCE veto - all dead) did not?

This script is the expensive GPU pass ONLY: it writes per-trade forecast scores.
Evaluation (coverage-matched veto metrics, neg-control, leakage split) lives in
kronos_veto_eval.py.

No look-ahead: context = 15m bars labeled <= dt1+45. Bars are left-labeled, so
the last context bar closes exactly at dt1+60 = trade entry. The four forecast
bars (labels dt1+60..dt1+105) span the 1h hold; the 4th bar's close = exit time.
Alignment is asserted per trade (last context label == dt1+45).

Pre-registered spec, fixed before the run (NO sweeps): lookback=480 (skip if
unavailable), pred_len=4, T=1.0, top_p=0.9, R=30 samples. p_up is computed from
30 independent sample paths (predict_batch averages internally over
sample_count, so each trade is duplicated R times with sample_count=1 -
compute-identical, see model/kronos.py auto_regressive_inference).

Exploratory only - NO Gauntlet verdict authority.

Usage:  python scripts/research/kronos_veto_score.py [--batch 12] [--limit N]
Resumable: appends to data/research/kronos_veto/scores.csv per chunk; on
restart, already-scored (ticker, dt1) pairs are skipped.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # Windows: no symlink privilege

import pandas as pd  # import BEFORE torch: lazy pandas-after-CUDA segfaults on Windows
import numpy as np
import torch

KRONOS_DIR = r"C:\Users\loq\Desktop\Trading\Kronos"
sys.path.insert(0, KRONOS_DIR)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
CACHE = os.path.join(ROOT, "data", "raw_upstox_cache_15min_3y")
OUT_DIR = os.path.join(ROOT, "data", "research", "kronos_veto")
OUT = os.path.join(OUT_DIR, "scores.csv")

LOOKBACK = 480
PRED_LEN = 4
R = 30            # independent sample paths per trade
T_SAMP = 1.0
TOP_P = 0.9
HF_MODEL = "NeoQuasar/Kronos-base"  # user goal 2026-07-03: base (102.3M), amended pre-run from small
HF_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


CUTOFF = pd.Timestamp("2025-09-09")  # HF weight upload date; post-cutoff = honest primary window


def load_trades():
    """Distinct (ticker, dt1) contexts, POST-cutoff first, seeded-random within window.

    Random order makes any prefix of the post-cutoff queue an unbiased subsample,
    so a pre-declared interim eval checkpoint is statistically valid (just lower
    power). Alphabetical order (the original) made partial data unreadable.
    """
    panel = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir"], parse_dates=["dt1"])
    trades = panel[["ticker", "dt1"]].drop_duplicates()
    trades["sym"] = trades["ticker"].str.replace(".NS", "", regex=False)
    post = trades[trades["dt1"] >= CUTOFF].sample(frac=1.0, random_state=42)
    pre = trades[trades["dt1"] < CUTOFF].sample(frac=1.0, random_state=42)
    return pd.concat([post, pre]).reset_index(drop=True)


def load_cache(sym):
    fp = os.path.join(CACHE, f"{sym}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=12, help="trades per GPU call (effective batch = batch*R)")
    ap.add_argument("--limit", type=int, default=0, help="score only first N trades (debug)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    trades = load_trades()
    print(f"panel: {len(trades)} distinct (ticker, dt1) trade contexts")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, parse_dates=["dt1"])
        done = set(zip(prev["ticker"], prev["dt1"]))
        print(f"resume: {len(done)} already scored")
    todo = trades[[(t, d) not in done for t, d in zip(trades["ticker"], trades["dt1"])]]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        print("nothing to do")
        return

    tokenizer = KronosTokenizer.from_pretrained(HF_TOKENIZER)
    model = Kronos.from_pretrained(HF_MODEL)
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

    caches, skips = {}, {"no_cache": 0, "no_align": 0, "short_ctx": 0, "nan_ctx": 0}
    rows_buf, n_chunks = [], 0
    batch_items = []  # (ticker, dt1, x_df, x_ts, y_ts, last_close)
    t_start = time.time()
    n_total = len(todo)
    n_done = 0

    def flush(items, chunk_idx):
        """One GPU call for len(items) trades, each duplicated R times, sample_count=1."""
        torch.manual_seed(1000 + chunk_idx)  # deterministic regardless of resume point
        df_list, xts_list, yts_list = [], [], []
        for (_, _, x_df, x_ts, y_ts, _) in items:
            for _ in range(R):
                df_list.append(x_df)
                xts_list.append(x_ts)
                yts_list.append(y_ts)
        preds = predictor.predict_batch(df_list, xts_list, yts_list, pred_len=PRED_LEN,
                                        T=T_SAMP, top_p=TOP_P, sample_count=1, verbose=False)
        out = []
        for i, (ticker, dt1, _, _, _, last_close) in enumerate(items):
            closes4 = np.array([preds[i * R + j]["close"].iloc[-1] for j in range(R)], dtype=float)
            rets = closes4 / last_close - 1.0
            out.append(dict(ticker=ticker, dt1=dt1, p_up=float((rets > 0).mean()),
                            mean_ret=float(rets.mean()), std_ret=float(rets.std()),
                            last_close=float(last_close), n_samples=R))
        return out

    def write_rows(rows):
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)

    # iterate in QUEUE order (post-cutoff random first) - do NOT regroup by ticker,
    # that would undo the randomization that makes interim checkpoints unbiased.
    for tr in todo.itertuples(index=False):
        sym = tr.sym
        if sym not in caches:
            caches[sym] = load_cache(sym)  # kept for the whole run (~few hundred MB total)
        df = caches[sym]
        if df is None:
            skips["no_cache"] += 1
            continue
        ts = df["timestamp"].values
        ctx_end_label = tr.dt1 + pd.Timedelta(minutes=45)
        pos = np.searchsorted(ts, np.datetime64(ctx_end_label), side="right") - 1
        if pos < 0 or ts[pos] != np.datetime64(ctx_end_label):
            skips["no_align"] += 1  # missing bar at dt1+45: alignment not provable, skip
            continue
        if pos + 1 < LOOKBACK:
            skips["short_ctx"] += 1
            continue
        ctx = df.iloc[pos + 1 - LOOKBACK: pos + 1]
        x_df = ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        if x_df.isnull().values.any():
            skips["nan_ctx"] += 1
            continue
        y_ts = pd.Series([ctx_end_label + pd.Timedelta(minutes=15 * (k + 1)) for k in range(PRED_LEN)])
        batch_items.append((tr.ticker, tr.dt1, x_df,
                            ctx["timestamp"].reset_index(drop=True), y_ts,
                            ctx["close"].iloc[-1]))
        if len(batch_items) >= args.batch:
            rows_buf.extend(flush(batch_items, n_chunks))
            n_done += len(batch_items)
            batch_items = []
            n_chunks += 1
            if len(rows_buf) >= 120:
                write_rows(rows_buf)
                rows_buf = []
            if n_chunks % 10 == 0:
                el = time.time() - t_start
                eta = el / max(n_done, 1) * (n_total - n_done)
                print(f"chunk {n_chunks}: {n_done}/{n_total} trades, {el/60:.1f}m elapsed, ETA {eta/60:.0f}m", flush=True)

    if batch_items:
        rows_buf.extend(flush(batch_items, n_chunks))
        n_done += len(batch_items)
    if rows_buf:
        write_rows(rows_buf)

    print(f"DONE: scored {n_done} trades in {(time.time()-t_start)/60:.1f}m; skips={skips}")
    n_out = len(pd.read_csv(OUT))
    print(f"scores.csv rows: {n_out} (incl. previous runs)")


if __name__ == "__main__":
    main()
