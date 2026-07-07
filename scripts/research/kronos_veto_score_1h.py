"""1h-native Kronos veto scores for the dualtf trade panel (base OR LoRA adapter).

Sibling of kronos_veto_score.py (which scores at 15-min). This one scores at the
model's native 1-hour resolution so a 1h-fine-tuned LoRA adapter is evaluated on the
resolution it was trained on. Timing mirrors the validated 15-min scorer exactly:

  entry = dt1 + 60min, hold = dt1+60 .. dt1+120 (one 1h bar).
  context = LOOKBACK 1h bars whose LAST bar is labeled dt1 (left-labeled => it covers
    dt1..dt1+60 and closes exactly at entry; known at entry, no look-ahead).
  forecast = 1 bar labeled dt1+60 (covers the hold hour); its close = exit price.
  p_up = fraction of R sampled paths whose forecast close > entry close.

This aligns with the panel's next-hour return `nhr`, so kronos_veto_eval.py can merge
these scores on (ticker, dt1) unchanged.

Bars are filtered to native full hours 09:15..14:15 (the 15:15 stub is dropped) to match
how the LoRA corpus was built (train == serve).

Exploratory only - NO Gauntlet verdict authority.

Usage:
  python scripts/research/kronos_veto_score_1h.py --tag base
  python scripts/research/kronos_veto_score_1h.py --tag lora \
        --adapter "C:/Users/loq/Desktop/Trading/kronos-nse-lora/adapters/1h"
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import pandas as pd  # before torch (Windows)
import numpy as np
import torch

KRONOS_DIR = r"C:\Users\loq\Desktop\Trading\Kronos"
sys.path.insert(0, KRONOS_DIR)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
CACHE = os.path.join(ROOT, "data", "raw_upstox_cache_1h_v3")
OUT_DIR = os.path.join(ROOT, "data", "research", "kronos_veto")

LOOKBACK = 240        # 1h context bars (~40 trading days)
PRED_LEN = 1
R = 30
T_SAMP = 1.0
TOP_P = 0.9
HF_MODEL = "NeoQuasar/Kronos-base"
HF_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
CUTOFF = pd.Timestamp("2025-09-09")


def load_trades():
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
    df = pd.read_csv(fp)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(timestamp=ts)
    df = df[(ts.dt.minute == 15) & (ts.dt.hour.between(9, 14))]   # native full hours
    return df.sort_values("timestamp").reset_index(drop=True)


def build_predictor(adapter):
    tok = KronosTokenizer.from_pretrained(HF_TOKENIZER)
    mdl = Kronos.from_pretrained(HF_MODEL)
    if adapter:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, adapter)
        print(f"[1h-veto] loaded LoRA adapter: {adapter}")
    return KronosPredictor(mdl, tok, device="cuda:0", max_context=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output suffix, e.g. base | lora")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir (omit for zero-shot base)")
    ap.add_argument("--batch", type=int, default=12, help="trades per GPU call (effective = batch*R)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--post-only", action="store_true", help="score only the post-cutoff honest window")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"scores_1h_{args.tag}.csv")

    trades = load_trades()
    if args.post_only:
        trades = trades[trades["dt1"] >= CUTOFF].reset_index(drop=True)
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out, parse_dates=["dt1"])
        done = set(zip(prev["ticker"], prev["dt1"]))
        print(f"resume: {len(done)} already scored")
    todo = trades[[(t, d) not in done for t, d in zip(trades["ticker"], trades["dt1"])]]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"panel: {len(trades)} contexts; todo {len(todo)}")
    if todo.empty:
        return

    predictor = build_predictor(args.adapter)
    caches, skips = {}, {"no_cache": 0, "no_align": 0, "short_ctx": 0, "nan_ctx": 0}
    rows_buf, batch_items, n_chunks, n_done = [], [], 0, 0
    t0 = time.time()

    def flush(items, chunk_idx):
        torch.manual_seed(1000 + chunk_idx)
        dfs, xts, yts = [], [], []
        for (_, _, x_df, x_ts, y_ts, _) in items:
            dfs += [x_df] * R
            xts += [x_ts] * R
            yts += [y_ts] * R
        preds = predictor.predict_batch(dfs, xts, yts, pred_len=PRED_LEN, T=T_SAMP,
                                        top_p=TOP_P, sample_count=1, verbose=False)
        out_rows = []
        for i, (ticker, dt1, _, _, _, last_close) in enumerate(items):
            closes = np.array([preds[i * R + j]["close"].iloc[-1] for j in range(R)], dtype=float)
            rets = closes / last_close - 1.0
            out_rows.append(dict(ticker=ticker, dt1=dt1, p_up=float((rets > 0).mean()),
                                 mean_ret=float(rets.mean()), std_ret=float(rets.std()),
                                 last_close=float(last_close), n_samples=R))
        return out_rows

    def write_rows(rows):
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)

    for tr in todo.itertuples(index=False):
        if tr.sym not in caches:
            caches[tr.sym] = load_cache(tr.sym)
        df = caches[tr.sym]
        if df is None:
            skips["no_cache"] += 1
            continue
        ts = df["timestamp"].values
        pos = np.searchsorted(ts, np.datetime64(tr.dt1), side="right") - 1
        if pos < 0 or ts[pos] != np.datetime64(tr.dt1):
            skips["no_align"] += 1
            continue
        if pos + 1 < LOOKBACK:
            skips["short_ctx"] += 1
            continue
        ctx = df.iloc[pos + 1 - LOOKBACK: pos + 1]
        x_df = ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        if x_df.isnull().values.any():
            skips["nan_ctx"] += 1
            continue
        y_ts = pd.Series([tr.dt1 + pd.Timedelta(minutes=60)])
        batch_items.append((tr.ticker, tr.dt1, x_df, ctx["timestamp"].reset_index(drop=True),
                            y_ts, ctx["close"].iloc[-1]))
        if len(batch_items) >= args.batch:
            rows_buf.extend(flush(batch_items, n_chunks))
            n_done += len(batch_items)
            batch_items = []
            n_chunks += 1
            if len(rows_buf) >= 120:
                write_rows(rows_buf)
                rows_buf = []
            if n_chunks % 10 == 0:
                el = time.time() - t0
                eta = el / max(n_done, 1) * (len(todo) - n_done)
                print(f"chunk {n_chunks}: {n_done}/{len(todo)}, {el/60:.1f}m, ETA {eta/60:.0f}m", flush=True)

    if batch_items:
        rows_buf.extend(flush(batch_items, n_chunks))
        n_done += len(batch_items)
    if rows_buf:
        write_rows(rows_buf)
    print(f"DONE: scored {n_done} in {(time.time()-t0)/60:.1f}m; skips={skips}; wrote {out}")


if __name__ == "__main__":
    main()
