"""Zero-shot Kronos-base next-day direction scan across the full 172-ticker
universe, once per trading day over the past month.

Research question (user ask, 2026-07-03): for every ticker and every trading
day in the past month, what did Kronos-base's zero-shot forecast say about
the NEXT trading day's direction, and how many predictions were UP vs DOWN?

This is a predicted-direction tally only -- it does NOT score against actual
outcomes (that would be a separate accuracy study). No veto/trading decision
is made here.

Convention: context = last LOOKBACK daily bars ending at trading day D-1's
close (bars strictly before D); forecast pred_len=1 -> the predicted return
is for day D itself (close-to-close). p_up = fraction of R sampled paths
whose predicted close is above the last context close.

Exploratory only -- NO Gauntlet verdict authority.

Usage: python scripts/research/kronos_daily_direction_scan.py
Resumable: appends to data/research/kronos_daily_scan/predictions.csv;
already-scored (ticker, day) pairs are skipped on restart.
"""
import os
import sys
import time
from datetime import date

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KRONOS_DIR = r"C:\Users\loq\Desktop\Trading\Kronos"
sys.path.insert(0, KRONOS_DIR)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

sys.path.append(os.getcwd())
from scripts.tickers import TICKERS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(ROOT, "data", "raw_daily_10y")
OUT_DIR = os.path.join(ROOT, "data", "research", "kronos_daily_scan")
OUT = os.path.join(OUT_DIR, "predictions.csv")

LOOKBACK = 400
PRED_LEN = 1
R = 10
T_SAMP = 1.0
TOP_P = 0.9
HF_MODEL = "NeoQuasar/Kronos-base"
HF_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
BATCH_ITEMS = 20  # ticker-day items per GPU call (x R paths each)

START_DATE = "2026-06-03"
END_DATE = "2026-07-02"
NSE_HOLIDAYS = {"2026-06-26"}


def trading_calendar():
    days = pd.bdate_range(START_DATE, END_DATE)
    return [d for d in days if d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS]


def load_cache(sym):
    fp = os.path.join(CACHE_DIR, f"{sym}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    calendar = trading_calendar()
    print(f"universe: {len(TICKERS)} tickers, {len(calendar)} trading days "
          f"({calendar[0].date()} .. {calendar[-1].date()})")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, parse_dates=["day"])
        done = set(zip(prev["ticker"], prev["day"]))
        print(f"resume: {len(done)} already scored")

    tokenizer = KronosTokenizer.from_pretrained(HF_TOKENIZER)
    model = Kronos.from_pretrained(HF_MODEL)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"loaded {HF_MODEL} on {device}")

    skips = {"no_cache": 0, "no_bar": 0, "short_ctx": 0, "nan_ctx": 0}
    rows_buf, n_chunks = [], 0
    batch_items = []  # (ticker, day, x_df, x_ts, y_ts, last_close, actual_close)
    t_start = time.time()
    n_done_before = len(done)

    def flush(items, chunk_idx):
        torch.manual_seed(2000 + chunk_idx)
        df_list, xts_list, yts_list = [], [], []
        for (_, _, x_df, x_ts, y_ts, _, _) in items:
            for _ in range(R):
                df_list.append(x_df)
                xts_list.append(x_ts)
                yts_list.append(y_ts)
        preds = predictor.predict_batch(df_list, xts_list, yts_list, pred_len=PRED_LEN,
                                        T=T_SAMP, top_p=TOP_P, sample_count=1, verbose=False)
        out = []
        for i, (ticker, day, _, _, _, last_close, actual_close) in enumerate(items):
            closes = np.array([preds[i * R + j]["close"].iloc[-1] for j in range(R)], dtype=float)
            rets = closes / last_close - 1.0
            p_up = float((rets > 0).mean())
            pred_dir = "UP" if p_up > 0.5 else ("DOWN" if p_up < 0.5 else "TIE")
            actual_ret = (actual_close / last_close - 1.0) if actual_close is not None else None
            actual_dir = None
            if actual_ret is not None:
                actual_dir = "UP" if actual_ret > 0 else ("DOWN" if actual_ret < 0 else "FLAT")
            out.append(dict(ticker=ticker, day=day, p_up=round(p_up, 4), pred_dir=pred_dir,
                            mean_ret=round(float(rets.mean()), 6), last_close=round(float(last_close), 4),
                            actual_close=actual_close, actual_dir=actual_dir, n_samples=R))
        return out

    def write_rows(rows):
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)

    all_pairs = [(t, d) for t in TICKERS for d in calendar]
    todo = [(t, d) for (t, d) in all_pairs if (t, d) not in done]
    n_total = len(todo)
    print(f"todo: {n_total} ticker-day forecasts")

    caches = {}
    for ticker, day in todo:
        sym = ticker.replace(".NS", "")
        if sym not in caches:
            caches[sym] = load_cache(sym)
        df = caches[sym]
        if df is None:
            skips["no_cache"] += 1
            continue
        ts = df["timestamp"].values
        idx = np.searchsorted(ts, np.datetime64(day))
        if idx >= len(ts) or ts[idx] != np.datetime64(day):
            skips["no_bar"] += 1
            continue
        if idx < LOOKBACK:
            skips["short_ctx"] += 1
            continue
        ctx = df.iloc[idx - LOOKBACK: idx]
        x_df = ctx[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)
        if x_df.isnull().values.any():
            skips["nan_ctx"] += 1
            continue
        x_ts = ctx["timestamp"].reset_index(drop=True)
        y_ts = pd.Series([pd.Timestamp(day)])
        last_close = float(ctx["close"].iloc[-1])
        actual_close = float(df["close"].iloc[idx])
        batch_items.append((ticker, day, x_df, x_ts, y_ts, last_close, actual_close))

        if len(batch_items) >= BATCH_ITEMS:
            rows_buf.extend(flush(batch_items, n_chunks))
            batch_items = []
            n_chunks += 1
            if len(rows_buf) >= 100:
                write_rows(rows_buf)
                rows_buf = []
            if n_chunks % 10 == 0:
                n_done = len(done) - n_done_before + n_chunks * BATCH_ITEMS
                el = time.time() - t_start
                eta = el / max(n_done, 1) * (n_total - n_done)
                print(f"chunk {n_chunks}: ~{n_done}/{n_total}, {el/60:.1f}m elapsed, ETA {eta/60:.1f}m", flush=True)

    if batch_items:
        rows_buf.extend(flush(batch_items, n_chunks))
    if rows_buf:
        write_rows(rows_buf)

    print(f"DONE in {(time.time()-t_start)/60:.1f}m; skips={skips}")
    n_out = len(pd.read_csv(OUT))
    print(f"predictions.csv rows: {n_out}")


if __name__ == "__main__":
    main()
