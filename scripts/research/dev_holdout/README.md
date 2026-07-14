# Dev / Holdout Gate Framework

Research gate & constraint hypotheses on a **development** window, then confirm each on a
**pre-registered, single-use holdout** — so the strategy layer can't quietly re-overfit the
way the "+26 bps / 13×" config did.

## The premise (why this split is the right one)

The v20 model is a **fixed, already-held-out input**. Its trainer
(`scripts/training/train_ranking_clean.py`) does a strict 80/20 temporal split:

```
train = 2022-01 .. 2025-06   |   val = 2025-07   |   UNTOUCHED test = 2025-08 -> onward
```

Everything from **2025-08 on is genuinely out-of-sample for the model**, and its rank-IC is
stable there (ρ≈0.02 short & long) — it does *not* degrade on Jun–Jul 2026. The model
generalizes. What overfit was the **gate layer** (thresholds hand-tuned to the backtest
window). So the model is frozen and the *gates* are what get the dev/holdout discipline.

## Split

| window | dates | use |
|---|---|---|
| **DEV** | 2025-08-01 → `HOLDOUT_START` | iterate/sweep/A-B gates freely |
| **HOLDOUT** | `HOLDOUT_START` → end (Jul 10 2026) | confirm once per hypothesis |

`HOLDOUT_START` defaults to **2026-06-11** (~1 month / ~22 trading days holdout). It is the
single most important pre-registration decision — change it with `--holdout-start YYYY-MM-DD`
but commit before you look.

## Files

- `build_feed.py` — scores the panel once with the frozen models → `data/research/dev_holdout/feed.parquet` (compact, offline, deterministic). Re-run with `--rebuild` to refresh S&P/rescore.
- `strategy.py` — config-driven 1-slot gate engine (faithful generalization of `scripts/backtests/temp_11m_combined.py`) + honest metrics.
- `run.py` — CLI: DEV runs free; HOLDOUT requires `--confirm --hypothesis`, logged to `HOLDOUT_LEDGER.md`.
- `configs/baseline.json` — the shipped 213-gate config.
- `HOLDOUT_LEDGER.md` — append-only record of every holdout look (the anti-peek trail).

## Usage

```bash
# 0. build the frozen feed once
python scripts/research/dev_holdout/build_feed.py

# 1. develop on DEV (free, iterate all you like)
python scripts/research/dev_holdout/run.py --config configs/baseline.json --neg-control 25 --monthly

# 2. a new hypothesis = a config diff, e.g. cap short conviction (inverted-U finding):
#    copy baseline.json -> myidea.json, set "short_conv_cap": 0.04, then run on DEV.

# 3. confirm ONCE on the holdout (records to the ledger; do not tune afterward)
python scripts/research/dev_holdout/run.py --config configs/myidea.json \
      --confirm --hypothesis "cap short conviction at 0.04 (inverted-U); expect short net > 0 OOS"
```

## Metrics discipline

- **PRIMARY = net bps/trade + t-stat**, per side. Leverage-independent and honest.
- **5× compound P&L is reported but labelled LEVERAGE-DEPENDENT** — it's the same net-bps
  stream on a 5×-reinvested base, not extra edge. The "13×" headline is this artifact.
- **Neg-control** (`--neg-control K`): random pick inside the same gated pool. If a config's
  edge over the neg-control is ~0, the gates — not the model's ranking — are doing the work,
  and gate edge is exactly what overfits. (Baseline DEV: +25.98 vs neg-control +21.48 → only
  +4.50 is ranking.)

## Rules (hard-won)

1. **Iterate on DEV only.** The holdout is single-use *per hypothesis*. Peek, then tune, then
   re-peek = you've made the holdout a second training set. That is the original failure, one
   level up.
2. **Pre-register.** State the hypothesis and expected direction *before* the holdout run.
3. **Prefer bps + t over rupees.** Rupees are leverage × compounding × window scale.
4. **A stable ρ≈0.02 model can't be turned into edge by thresholds.** Gates allocate/filter
   the signal; they don't add information. The winnable levers are sizing & cost, or new data.

## ⚠️ Data-integrity caveats (discovered building this)

- **Panel** `data/research/v20_rolling_1h/panel.parquet` was rebuilt mid-development: it now
  runs to **2026-07-10** with the earlier Feb 20–Mar 24 gap **filled**. The audited doc's exact
  figures (213 / +26.59) were produced on the *old gapped* panel; on the current panel the same
  config gives 218 / +25.98. Frozen-feed provenance is why this framework exists.
- **Nifty cache** `data/raw_index_cache/nifty50_15m.csv` is **double-convention corrupted**
  (rows ≤ ~2026-06-09 are IST wall-clock mislabelled `+0000`; rows ≥ ~2026-06-10 are true UTC;
  Jun 8–9 contain both). `build_feed.py` normalizes both to naive IST, but the cache should be
  **re-collected cleanly**. Until then, Jun 8–9 nifty (2 DEV days) may carry a few mis-gridded
  bars; the holdout (Jun 11+) is clean.
