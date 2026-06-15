---
title: "Conv 2026-06-15 — v20 Rolling-1h Model"
type: log
status: wip
updated: 2026-06-15
model: v20_rolling_1h
tags: [conversation, model, research, 1h, transformer-adjacent]
---
# 💬 Conversation Context: v20 Rolling-1h (Overlapping-Window) Model

## 📌 Metadata
- **Start Date**: 2026-06-15
- **Status**: 🔴 Concluded (build done; timing/pocket use is the open follow-up)
- **Focus Area**: Models — 1h ranker variant

## 🎯 Objectives
- [x] Build a v20 dataset: overlapping 1h candles stepped every 15 min ([09:15-10:15], [09:30-10:30] … [14:15-15:15]) instead of v10's 6 non-overlapping exchange hourly bars.
- [x] Train the v10 recipe on it (rank:pairwise, ndcg@3, depth 5, monthly purged WF).
- [x] Report an honest WF comparison vs v10 — **research only, no Gauntlet, ⚠️ UNVERIFIED**.

## ✅ Result (see [[04 — Research/V20 Rolling-1h Overlapping-Window Model|full write-up]])
- Backfilled 2022 15-min (Upstox V3 minutes/15; 2022 available, 2021 not) → panel rebuilt on full span: **3,022,168 rows / 17,671 cross-sections / 2022-01→2026-06**, 18 labeled entries/day. Models in `models/research/v20_rolling_1h/` (unregistered).
- WF (EQUAL SPAN 2022-2026): v20 Long ρ **0.0323** / Short ρ **0.0327**, L-WR@3 **54.3%** / S-WR@3 **53.6%** vs v10 0.0261/0.0245, 52.4%/53.6% → **v20 ranks ~24-33% better, both sides, steadier.**
- ⚠️ The first run was truncated 2023-2026 (pre-backfill) and showed v20≈v10 "within noise" — that was a **span artifact**; the equal-span result supersedes it.
- **BUT** gross edge ~4 bps/side/bar vs 10 bps cost → **still sub-cost standalone**. v20 = a **stronger FILTER/ranker, not a money-maker** (same category as v10 FILTER_GRADE).
- **Isolation test DONE** (`scripts/research/v20_isolation_eval.py`): on v10's 5 shared :15 moments v20 = L 0.0323 / S 0.0328 (≈ its all-18 avg) and beats v10 there by +0.0062 / +0.0083. → lift is the **construction**, not extra-moment cherry-picking. Confound resolved in v20's favor.
- **GAUNTLET CERTIFIED** (run `20260615T175149Z-5f7d069f`, dataset `1h_roll_15anchor` = non-overlapping :15 subset; overlapping panel can't be validly graded): **LONG + SHORT FILTER_GRADE**. vs v10's certified short = a PEER (short ρ ~0.029 vs 0.025; Top-1 short net −3.3 vs −6.3bps [v20 better]; Top-3 tied; both sub-cost). The "24-33% better" was inflated by overlap + easy-moment averaging. v20 = certified peer-FILTER, marginally stronger at top-of-book, NOT a tradeable standalone short. Stamped into model metadata; ledger updated.
- **Pending (user choice)**: full-overlapping-panel DIAGNOSTIC run (non-authoritative) to illustrate the significance inflation — optional given the cert already settled the question.

## 💻 Active Code Files
- [build_rolling_1h_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/build_rolling_1h_panel.py) — new panel builder (aggregates rolling 1h from `data/raw_upstox_cache_15min_3y/`, no API).
- [train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py) — added `--tf 1h_roll` config + parquet support.

## 📝 Compacted Session Log
- **Premise**: user wants a new 1h model "based on v10" but trained on rolling/overlapping 1h windows (15-min step) — the "walk-forward constructed candles".
- **Decisions (user)**: label = **next 1h** (mirrors v10 `Next_Hour_Return`); purpose = **research/exploration** (no verdict authority).
- **Construction**: rolling window = 4 consecutive native 15-min bars; keyed by entry time T (= hour close); label `close(T+1h)/close(T)-1` via exact-timestamp reindex → auto session-mask. Smoke test (AARTIIND): 22 windows/day, **18 labeled/day** (10:15–14:30) vs v10's 5. Feature/z-score schema reused VERBATIM from `collect_upstox_1h_v3.py` → identical 86-feature set.
- **⚠️ Methodology caveat (logged in code)**: overlapping windows share 45/60 min → consecutive rows ~75% autocorrelated. Monthly WF point estimates (avg ρ, WR@k) are comparable to v10, but effective N ≈ ¼ of rows → **do NOT t-test / read significance**. Prior (4 independent lines) says 1h price/volume edge is information-limited, not data-size limited; expectation is similar per-trade edge to v10, value is **signal frequency / anchor-agnostic serving**, not better alpha.
- **Lookback nuance**: compute_features uses fixed BAR-COUNT windows, so on the 15-min-spaced grid TA lookback is ~4× shorter in wall-clock than v10. Inherent; flagged.
- **Span caveat**: 15-min cache spans 2023-01→2026-06 (~41 mo) vs v10's 2022→2026 (54 mo) → fewer WF folds; not a span-controlled comparison.

## 🔗 Core Memory Links
- Host model: [[02 — Models/_Shared/Model Performance & Statistics|Model Performance & Statistics]] (v10_native_1h)
- Builds on: [[02 — Models/_Shared/Model Registry & File Structures|Model Registry]]
