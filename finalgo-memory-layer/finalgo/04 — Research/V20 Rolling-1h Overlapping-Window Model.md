---
title: "V20 Rolling-1h Overlapping-Window Model"
type: research
status: concluded
updated: 2026-06-15
model: v20_rolling_1h
tags: [research, 1h, ranker, overlapping-windows, anchor-agnostic]
---
# V20 Rolling-1h Overlapping-Window Model

> [!warning] ⚠️ UNVERIFIED — research only, NO Gauntlet run, no verdict authority.
> All numbers below are walk-forward **point estimates** from the training harness
> ([train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py) `--tf 1h_roll`).
> Overlapping windows make consecutive rows ~75% autocorrelated → effective N ≈ ¼ of row
> count, so **significance/CI are inflated; do NOT t-test these.** Not in `models/registry.json`.

## Hypothesis
Train the v10 recipe on **overlapping rolling 1h candles stepped every 15 min**
([09:15-10:15], [09:30-10:30], … instead of v10's 6 non-overlapping exchange hourly bars),
to get a 1h-horizon ranking that can be **refreshed every 15 min** (anchor-agnostic serving).

## Build (no new data collection)
- A rolling 1h window = 4 consecutive native 15-min bars from `data/raw_upstox_cache_15min_3y/`.
  Builder: [build_rolling_1h_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_rolling_1h_panel.py).
- Feature / z-score / Relative_* pipeline copied **verbatim** from `collect_upstox_1h_v3.py` →
  identical **86-feature** schema to v10. Only the candle grid differs.
- Label `Next_Hour_Return = close(T+1h)/close(T)-1`, exact-timestamp reindex = session mask.
- 2022 15-min was backfilled into the cache (Upstox V3 minutes/15; probe confirmed 2022 available,
  2021 not) so the panel matches v10's span. Builder `scripts/research/backfill_2022_15min.py`.
- Panel (full span): `data/research/v20_rolling_1h/panel.parquet` — **3,022,168 rows, 17,671
  cross-sections**, 171 tickers/query, **18 labeled entry-times/day** (10:15–14:30) vs v10's 5.
  Span **2022-01→2026-06 (52 months)**.

## Result (purged monthly walk-forward) — EQUAL SPAN
| WF avg | **v20 rolling-1h** (8 folds) | v10 native-1h `b10b37fc…` (9 folds) |
| --- | --- | --- |
| Long ρ | **0.0323** | 0.0261 |
| Short ρ | **0.0327** | 0.0245 |
| Long WR@3 | **54.3%** | 52.4% |
| Short WR@3 | 53.6% | 53.6% |
| Long / Short gross edge | +4.1 / +3.9 bps per bar | (sub-cost) |
| Span | 2022–2026 | 2022–2026 |

v20 per-fold Long ρ: 0.030, 0.043, 0.043, 0.024, 0.040, 0.025, 0.038, 0.016 (last fold 2026-02→03).
> Earlier truncated run (2023→2026 only, before 2022 backfill) gave v20 ρ ≈ 0.027 ≈ v10 — that
> "within noise" read was a **span artifact** (it omitted v10's strong 2022-trained folds). Corrected.

## Verdict (research)
- **On equal footing, v20 is a moderately BETTER RANKER than v10** — ~24–33% higher WF ρ on both
  sides, steadier across folds. This is **not** an autocorrelation artifact (overlap inflates
  precision/CI, not the mean ρ). Revises the initial "≈ v10" read.
- **BUT still sub-cost as a standalone trade:** ~4 bps/side/bar gross vs the 10 bps binding cost →
  net-negative alone. v20 is a **stronger filter/ranker, not a money-maker** — same category as v10
  (FILTER_GRADE). Would need a Gauntlet run to certify any filter uplift (not done; research only).
- **Construction confirmed as the source (isolation test, `scripts/research/v20_isolation_eval.py`):**
  scored on **only v10's 5 shared :15 decision moments**, v20 = Long 0.0323 / Short 0.0328 —
  essentially identical to its all-18 average, and **beats v10 there by +0.0062 / +0.0083**. So the
  off-:15 moments are NOT secretly easier; the lift is the rolling-feature construction + 3.6× more
  training cross-sections, not moment-cherry-picking. (Construction includes the data-volume gain;
  not separated from feature-lookback, but both are inherent benefits of the approach.)
- **Caveat (lookback):** `compute_features` uses fixed BAR-COUNT windows, so on the 15-min-spaced
  grid TA lookback is ~4× shorter in wall-clock than v10. Inherent to the construction.

## ✅ Gauntlet certification — 2026-06-15 (`run 20260615T175149Z-5f7d069f`)
Certified on the **non-overlapping :15 subset** (`1h_roll_15anchor`, v10's cadence) — the overlapping
18/day panel **cannot be validly graded** (overlap → autocorrelated queries → inflated CI/t-stats). 5
entries/day, 839,391 rows, Stage-0 passed (79.99% verified / 20% boundary = day's-last-entry, waiver +
anti-overnight check). Deflated t-threshold 1.96 (fresh dataset family, 0 priors).

- **Verdict: LONG `FILTER_GRADE`, SHORT `FILTER_GRADE`** — same grade as v10.
- **vs v10's certified short (`20260610T184210Z`): a PEER, not a clear winner.** Short ρ modestly
  higher (~0.029 vs ~0.025); Top-1 short net @10bps **−3.3 bps (t −1.2) vs v10 −6.3 (t −3.8)** (v20
  better, no longer significantly negative); Top-3 short net @10bps **−7.8 vs −7.7** (tied). **Both
  sub-cost.** The earlier "24–33% better" was inflated by the overlapping panel + easy-moment averaging;
  on the clean cert the edge is marginal and concentrated at k=1.
- **Net:** v20 = a *certified* short FILTER, marginally stronger than v10 at the top of the book, still
  **not a tradeable standalone short.** Stamped into `models/research/v20_rolling_1h/metadata.json`.

## Reusable artifacts
- Panel: `data/research/v20_rolling_1h/panel.parquet` (gitignored)
- Model: `models/research/v20_rolling_1h/` (XGB long/short, not registered)
- Rerun: `python scripts/research/build_rolling_1h_panel.py` then
  `python scripts/training/train_ranking_clean.py --tf 1h_roll`

## Links
- Host model: [[02 — Models/_Shared/Model Performance & Statistics|v10_native_1h]]
- Confirms ceiling line: [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]]
- Session: [[06 — Logs/Conversations/Conv-2026-06-15-v20-Rolling-1h-Model|Conversation Log]]
