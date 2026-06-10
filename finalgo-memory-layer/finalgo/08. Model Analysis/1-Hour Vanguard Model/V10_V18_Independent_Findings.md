# V10 + V18 — Independent Findings (Claude, OOS 2025-07 → 2026-06)

> Independent re-evaluation built from raw data + saved model artifacts, **not** a re-run of the
> author's backtest scripts. Reproducible via `scripts/analysis/v10_v18_independent_analysis.py`
> (+ `v10_v18_walkforward.py`) and `notebooks/v10_v18_independent_analysis.ipynb`.
> Every number below is on the **clean** OOS window unless noted; costs shown at **6 bps** (user's
> real all-in cost) and **10 bps** (slippage-padded threshold). t = t-stat on mean net return/trade.

This document **supersedes and corrects** `V10_vs_V18_Edge_Evaluation.md` (see §6).

---

## 0. BOTTOM LINE (read this first)

**The edge is not deployable. The saved models' OOS performance is NOT reproducible.**

The saved `v10` artifacts show a strong recent-OOS edge (+11 to +15 bps raw). But when the model
is **retrained** — whether by rolling walk-forward or by reproducing the exact documented recipe
from scratch (same data window, same params, same seed, same early-stopping) — **the edge
disappears**: raw returns fall to ~0–4 bps and net is **negative** at the user's 6 bps cost across
every config (V10-alone, hybrid, asymmetric). The short ranker even early-stops at *iteration 0*
(no learnable signal on validation).

I could not reproduce the saved short model's +12.6 bps raw on the exact same 6 months — a faithful
rebuild yields **−0.4 bps**. A 10+ bps gap from a model that is supposedly trained the documented
way means the **saved artifacts came from a different process than documented — most likely data
leakage in their training run, or an undocumented/over-fit fit.** Since any live system retrains,
you would get the negative-net reality, not the saved-model backtest.

**Action: do not deploy on these models. Audit the provenance of `models/v10_native_1h/` (was it
trained only on ≤2025-06, or did test data leak in?). Sections 2–4 below describe the saved-model
behavior; §5 is the reproducibility failure that invalidates it for deployment.**

---

## 1. Data integrity (Phase 0) — the earlier "overnight artifact" alarm was WRONG

I initially flagged that the 13:15 last bar has a populated `Next_Hour_Return` with no 14:15
cross-section, and that 7.3% of returns exceed ±100 bps — the overnight-artifact signature from
memory. **Direct reconstruction from `Close` disproves it:**

| Hour | matches *overnight* close/close | % \|fwd\| > 100bps |
|------|-------------------------------|--------------------|
| 9–12 | 99.8–99.9% match *same-day next bar* | 5.8–9.7% |
| 13 (last) | **0.4%** (does NOT match overnight) | 8.6% |

The 13:15 target is a genuine **same-day 13:15→14:15** intraday return (to a bar simply not
included as a tradeable cross-section). Only **0.04%** of rows are genuine overnight contamination
(missing intermediate bars). The fat tail is **real open-hour (9:15) volatility**, not leakage.
→ The memory warning about overnight artifacts (`reference_ranking_data_conventions`) applied to
the dual-TF/15m work; it does **not** apply to this 1h target's construction.

> **Sections 2–4 describe the behavior of the SAVED model artifacts.** They are accurate for those
> frozen files but, per §5, that performance is **not reproducible** by retraining — so read them as
> "what the saved files do," not "what you can expect live."

## 2. V10 standalone (Phase 1) — a top-decile-only ranker, shorts > longs

- **Rank IC** degrades in-sample→OOS but stays positive: long 0.045→0.030, short 0.044→0.031.
- **Decile monotonicity FAILS.** OOS decile means (bps): `[-1.4, -1.6, 0.1, 0.3, 0.4, 1.3, 1.8, -0.6, -0.6, 4.5]`.
  Edge lives **only in the top decile** — the middle is noise. V10 is an *extremes detector*, not a
  full-distribution ranker. This is why Top-1/Top-3 work and Top-5+ dilute.
- **Top-K net (OOS clean):**

  | K | LONG @6 | LONG @10 | SHORT @6 | SHORT @10 |
  |---|---------|----------|----------|-----------|
  | 1 | +8.0 (t4.3) | +4.0 (t2.1) | +17.8 (t7.2) | +13.8 (t5.6) |
  | 3 | +5.0 (t5.1) | +1.0 (t1.0) | +9.3 (t7.0) | +5.3 (t4.0) |
  | 5 | +3.5 | −0.5 | +7.1 | +3.1 |
  | 10| +1.0 | −3.0 | +2.8 | −1.2 |

  **Shorts are the stronger standalone signal.** Longs alone barely clear 10 bps.

## 3. V18 standalone (Phase 2) — NO standalone edge; it's a weak clock

- **AUC overfits hard:** in-sample 0.648/0.630 → **OOS 0.517/0.516** (barely above coin-flip;
  the author's doc cites 0.524).
- **Trading every bar V18 green-lights loses money at every threshold** (net −9 to −11 bps @10bps;
  even at 6 bps it's negative). Reliability curve is shallow.
- **"Is it just a clock?"** V18-gated long net −6.6 bps (n=38k) ≈ the naive *"trade only hour 13"*
  baseline (−8.7). V18 adds essentially no directional information beyond a time-of-day filter.

**So how does the hybrid work?** V18 is useless *alone* but, applied as a **veto on V10's already-
ranked top-3 longs**, it selects a better subset (conditional value, not standalone alpha).

## 4. Combined (Phase 3) — the asymmetric config wins

**Veto marginal decomposition (OOS clean, Top-3, 10 bps):**

| Side | veto OFF | veto ON | verdict |
|------|----------|---------|---------|
| LONG | +1.0 (n3435) | **+4.6** (n1028) | veto **helps** — keep it |
| SHORT| **+5.3** (n3435) | +4.6 (n1543) | veto **hurts** — drop it |

**Config comparison (OOS clean, Top-3):**

| Config | LONG @6 | LONG @10 | SHORT @6 | SHORT @10 |
|--------|---------|----------|----------|-----------|
| V10 alone | +5.0 | +1.0 | +9.3 | +5.3 |
| Hybrid symmetric | +8.6 | +4.6 | +8.6 | +4.6 |
| **Asymmetric (veto longs, raw shorts)** | **+8.6 (t4.4)** | **+4.6 (t2.3)** | **+9.3 (t7.0)** | **+5.3 (t4.0)** |
| Dual-lock | +9.3 | +5.3 | +2.9 | +2.9 (t1.4) |

- **Dual-lock "100–115 bps" is debunked:** correctly computed per-trade net is **+2.9 bps** short
  (n=1215), *worse* than asymmetric. The heatmap figure was not a per-trade net return.
- **OOS-half stability (asymmetric @10bps):** LONG H2-25 +2.3 → H1-26 +6.8; SHORT H2-25 +2.9 →
  H1-26 +8.1. Edge present in **both** halves and **stronger** recently — no decay.

## 5. Walk-forward robustness (Phase 4) — THE REPRODUCIBILITY FAILURE

Genuinely-OOS rolling retrain of both models (`v10_v18_walkforward.py`), **matched to the production
recipe** (validation month + `early_stopping_rounds=50`). All net values @6 bps (user's real cost).

**Walk-forward, 2025-07+ (same recent regime where the saved model looked strong):**

| Config | LONG net | SHORT net | raw short |
|--------|----------|-----------|-----------|
| V10 alone | −6.0 (t−4.2) | −3.6 (t−1.9) | +2.4 |
| Hybrid symmetric | −3.5 | −2.1 (t−0.8) | +3.9 |
| Asymmetric | −3.5 | −3.6 | +2.4 |

Every config is **net-negative even at 6 bps, even on the recent regime**. (Full-period 2023-07+ is
similarly negative.)

**The decisive test — saved model vs faithful rebuild on the *exact same 6 OOS months*:**

| On the same 6 months, Top-3 @6 bps | LONG raw | LONG net | SHORT raw | SHORT net |
|------------------------------------|----------|----------|-----------|-----------|
| **Saved production artifact** | +13.0 | +7.0 | +12.6 | +6.6 |
| **From-scratch, documented recipe** (train ≤2025-06, val 2025-07, seed 42, early-stop 50) | +3.7 | −2.3 | **−0.4** | −6.4 |
| **Rolling walk-forward retrain** | — | −6.0 | +2.4 | −3.6 |

The saved model's edge **cannot be reproduced** from its own documented recipe (the short ranker
even early-stops at iteration 0 → no signal). A ~10–13 bps gap that vanishes when you rebuild the
model the documented way is the classic signature of **leakage or non-reproducible provenance** in
the saved artifact — not a robust trading edge.

> Methodology note: I first ran the WF without early stopping (a bug — production uses it). That
> first run also failed, but unfairly; I fixed it to match production and the failure persists, so
> the conclusion is sound. Separately, the production recipe uses **2025-07 as its early-stopping
> validation month**, which overlaps the OOS window everyone evaluated on — a real (if minor) leak
> in the original evaluation.

## 6. Corrections to `V10_vs_V18_Edge_Evaluation.md`

1. **Missing artifacts:** it cites `feature_importance.png`, `shap_summary.png`,
   `distributions_roc.png`, `bucket_returns.png` — none exist in the assets folder.
2. **"+9.61 bps V18-gated edge":** not reproducible. V18-gated trades are **net-negative** OOS
   (the ToD heatmap it references is red in every cell at 10 bps fees).
3. **"v18 isolates ~9.6 bps regimes":** false — V18 alone is net-negative and ≈ a clock.
4. **Unmentioned:** V18's AUC overfit (0.65→0.52), the veto **hurting** shorts, and the
   top-decile-only nature of V10.
5. The architecture conclusion (rank with V10, gate with V18) is **directionally right for longs
   only** — and the supporting magnitudes were inflated.

## 7. Verdict

**Do not deploy.** The headline edge exists only in the *saved model artifacts*, which I cannot
reproduce. Every honest retrain (rolling WF and from-scratch documented recipe) gives **negative net
at 6 bps**. What the saved-model backtest (§2–4) shows is real *for those frozen files*, but it is
not a property of the V10/V18 method — so it will not survive the retraining any live system does.

Ordered next steps:
1. **Audit `models/v10_native_1h/` provenance.** Confirm what data it was actually trained on. The
   irreproducibility + metadata `total_rows = 928,078` (the *full* 54-month dataset) raise a real
   leakage suspicion. If the saved model saw test data in training, that fully explains §5.
2. **Fix the evaluation leak:** never use an OOS month (2025-07) as the early-stopping validation set.
3. **Re-establish a baseline honestly** with the walk-forward harness here (`v10_v18_walkforward.py`)
   — that is the number you can trust. Currently it says: no net edge after costs.
4. Only after a clean retrain shows positive walk-forward net should configuration questions
   (asymmetric veto, K, thresholds) be revisited.

**What still holds regardless** (method-level, not vintage-dependent):
- V18 has no standalone directional edge (OOS AUC ≈0.52, net-negative at every threshold, ≈ a clock).
- V10 only ranks at the extreme top decile; K>3 dilutes.
- The data target is clean (no overnight artifact); costs of 6 bps are the right hurdle.
