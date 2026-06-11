# Cross-Sectional Transformer (CST) — Architecture Proposal & Research Spec

**Date:** 2026-06-10
**Status:** 📋 PROPOSAL — not built, not validated. All performance numbers in this document are targets or priors, **⚠️ UNVERIFIED** (no Gauntlet `run_id` exists).
**Author context:** Written after the v8–v19 demotion arc. See [[02. Model Suite/Model Performance & Statistics]].

---

## 1. Executive Summary

Every model trained so far (v8/v10 XGBoost rankers, v17/v18 Random Forest, v19 CatBoost, TBM-labeled variants) scores each ticker **in isolation** and ranks afterwards. All five paradigms converged to the same ceiling: raw cross-sectional Spearman ρ ≈ 0.025 (decaying to ~0.01), directional AUC ≈ 0.50, net-negative at 6 bps. This convergence implies the ceiling lives in the *information set*, not in model capacity — so swapping in a bigger model on the same inputs is pointless.

The Cross-Sectional Transformer (CST) is the one architecture that changes the information set **within price action**: at every timestamp it sees the entire universe (~172 tickers) simultaneously and lets each ticker attend to all others. This exposes information no per-ticker model has ever received:

- **Lead–lag**: index futures / sector leaders moving before laggards.
- **Peer divergence**: a ticker deviating from its correlated cluster.
- **Breadth / dispersion context**: how crowded or thin the current move is.

Supporting evidence that the cross-sectional dimension is where the remaining signal lives: the *only* statistically real surviving signal in the suite is cross-sectional — the short-side raw Top-3 win-rate (v8 z=5.4, v10 z=7.3, still z=3.8 in the last 12 months). See the Raw-Signal-Only Audit in [[02. Model Suite/Model Performance & Statistics]].

**Honest prior:** the most likely outcome is a modest lift (ρ ≈ 0.03–0.05) that still struggles against 6 bps costs. This proposal therefore includes a pre-registered kill-bar and a cheap Stage-0 falsification test that must run *before* any transformer code is written.

---

## 2. Problem Formulation

This is a **listwise learning-to-rank problem over a fixed universe**, not per-ticker forecasting.

- **Query**: one timestamp `t` (one completed 15m or 1h bar across the universe).
- **Documents**: the ~172 tickers active at `t`.
- **Label** per ticker: forward 1-bar return, transformed to its **cross-sectional z-score (or rank) within the query**. Never the absolute return — v18/v19 proved absolute direction is unlearnable from these features (AUC ≈ 0.50), while relative ordering carries the only real signal.
- **Output**: one scalar score per ticker per query; trade decision = top-K / bottom-K of the score vector.

This matches the production task exactly: "pick the best 2–3 longs and shorts out of 172 right now."

---

## 3. Input Representation

### 3.1 Per-query tensor

For each timestamp `t`:

```
X[t]  ∈  ℝ^(S × N × F)      S = ~172 tickers, N = lookback bars, F = features per bar
M[t]  ∈  {0,1}^S            validity mask (suspended / missing / illiquid names)
C[t]  ∈  ℝ^(G)              market-context vector (global token, see 3.3)
y[t]  ∈  ℝ^S                cross-sectional z-scored forward return
```

Recommended starting values: `N = 16` bars (≈ 2.3 trading days at 1h; ≈ 1 day at 15m), `F ≈ 48` (reuse the existing normalized feature pipeline).

### 3.2 Features — reuse the existing pipeline

The 48-feature normalization strategy in [[02. Model Suite/Feature Engineering & Normalization]] was *designed* for cross-sectional comparability (percentage distances, bounded oscillators, OBV rate-of-change) and is reused as-is, with two changes:

1. **Keep the cross-sectional z-scoring** (per query, across tickers) — it is exactly the right normalization for attention across tickers.
2. **Add raw-ish relative returns** at multiple lags (1, 2, 4, 8 bars), cross-sectionally z-scored. Lead–lag detection needs clean recent-return information per ticker; the existing indicator set partially buries it.

All features computed on **completed bars only** (the IBS same-bar leakage lesson — see [[06. Context & Logs/Daily Logs/2026-06-08|2026-06-08 diagnosis]]).

### 3.3 Static & context embeddings

- **Ticker embedding** (`S × d_emb`, learned): lets attention learn persistent pair relationships (e.g., HDFC Bank ↔ ICICI Bank lead-lag) beyond what features express. Risk: identity memorization — mitigated by ticker-dropout (§6).
- **Sector/industry embedding** (learned, from a static mapping): gives the model the correlation-cluster prior for free.
- **Time-of-day embedding**: one embedding per bar slot (7 slots at 1h, ~25 at 15m). Justified by the strong documented ToD structure (14:30 pocket on 1h; 14:00/14:15 on 15m).
- **Global market token**: one extra "ticker" in the attention sequence carrying market-level features — Nifty futures return (lagged), India VIX level/change, breadth (% of universe above VWAP / above open), cross-sectional dispersion of returns, day-of-week. This is the cheapest way to inject regime context and lets every ticker attend to "the market" directly.

---

## 4. Architecture

```
                       ┌─────────────────────────────────────────────┐
 per ticker s:         │  Temporal Encoder (shared weights, small)   │
 X[t,s] ∈ ℝ^(N×F) ───▶│  Linear(F→d) + 1-layer GRU  (or TCN/patch)  │──▶ h_s ∈ ℝ^d
                       └─────────────────────────────────────────────┘
                                  + ticker_emb(s) + sector_emb(s) + tod_emb(t)

 tokens = [h_1, …, h_S, g]        g = global market token (from C[t])

                       ┌─────────────────────────────────────────────┐
                       │  Cross-Sectional Encoder                    │
                       │  2–4 × TransformerEncoderLayer              │
                       │  (d_model=96, heads=4, ff=192, pre-norm,    │
                       │   attention ACROSS the S+1 tokens,          │
                       │   key_padding_mask = ¬M[t])                 │
                       └─────────────────────────────────────────────┘

                       ┌─────────────────────────────────────────────┐
                       │  Score Head: LayerNorm → Linear(d→1)        │
                       │  (optional twin heads: long-score,          │
                       │   short-score — mirrors the v8 two-model    │
                       │   long/short split)                         │
                       └─────────────────────────────────────────────┘
 output: score vector ∈ ℝ^S  →  rank → top-K / bottom-K
```

Key design decisions:

| Decision | Choice | Rationale |
|---|---|---|
| Where attention runs | **Across tickers**, not across time | Time is handled by the cheap temporal encoder; the novel information is cross-name. This is the iTransformer-style inversion. |
| Positional encoding across tickers | **None** | The universe is a *set*, not a sequence — the model must be permutation-equivariant over tickers. Identity comes from ticker embeddings, not positions. |
| Temporal encoder size | Tiny (1-layer GRU or linear patch embed) | N=16 is short; v8 proved most temporal info is already in the engineered features. Don't spend parameters here. |
| Model size | **d_model 64–128, 2–4 layers, ≈ 0.3–1.2 M params** | Sample size (§5) cannot support more. If a 1M-param model finds nothing, a 100M-param model finds noise. |
| Heads | 4 | Enough to specialize (sector head, market-token head, lead-lag head) without fragmenting d_model. |
| Long/short | Twin heads or two separate trainings | The raw-signal audit shows the short side carries nearly all the real signal; do not let a single head average it away with dead long-side gradients. |

### 4.1 PyTorch sketch (reference only)

```python
class CST(nn.Module):
    def __init__(self, F=48, d=96, n_layers=3, n_heads=4, S=172, n_sectors=20, n_tod=7):
        super().__init__()
        self.temporal = nn.GRU(F, d, batch_first=True)          # shared across tickers
        self.tick_emb = nn.Embedding(S, d)
        self.sec_emb  = nn.Embedding(n_sectors, d)
        self.tod_emb  = nn.Embedding(n_tod, d)
        self.glob_proj = nn.Linear(G_FEATURES, d)                # market token
        layer = nn.TransformerEncoderLayer(d, n_heads, 2*d, dropout=0.2,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, X, tick_id, sec_id, tod_id, g, mask):
        B, S, N, F = X.shape
        h = self.temporal(X.reshape(B*S, N, F))[1][-1].reshape(B, S, -1)
        h = h + self.tick_emb(tick_id) + self.sec_emb(sec_id) \
              + self.tod_emb(tod_id)[:, None, :]
        tokens = torch.cat([h, self.glob_proj(g)[:, None, :]], dim=1)
        pad = torch.cat([~mask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
        z = self.encoder(tokens, src_key_padding_mask=pad)[:, :S, :]
        return self.head(z).squeeze(-1)                          # (B, S) scores
```

---

## 5. Sample-Size Budget (why the model must stay small)

Under the 3-year mandate ([[02. Model Suite/Training Data & Regime Requirements]]), ~750 trading days:

| Timeframe | Queries (timestamps) | Ticker-rows (×172) | Verdict |
|---|---|---|---|
| 1h (~7 bars/day) | ~5,200 | ~900 K | Supports ≤ ~1 M params with heavy regularization |
| 15m (~25 bars/day) | ~19,000 | ~3.3 M | Supports the same model more comfortably |

The effective sample for a *listwise* model is the **query count**, not the row count — the Row Count Fallacy applies with full force. 5,200 queries is small; this is the central argument against any large transformer, MoE, or pre-trained-LLM-style approach. Calendar diversity, not rows, remains the binding constraint.

Recommendation: **train on 15m, evaluate on both 15m and 1h horizons** — 15m gives 3.6× more queries from the same calendar span at no regime-diversity cost.

---

## 6. Training Procedure

- **Loss**: primary **ListMLE** (listwise, rank-calibrated) over each query's valid tickers; auxiliary Huber regression on the cross-sectional z-scored forward return (weight ~0.3) to stabilize early training. Alternative if ListMLE is unstable at S=172: pairwise logistic loss on sampled within-query pairs with |Δrank| weighting (closest to the proven `rank:pairwise` v8 setup).
- **Batching**: one batch = a set of complete queries (e.g., 8 timestamps). Never mix tickers from different timestamps into one attention pass.
- **Optimizer**: AdamW, lr 3e-4 with cosine decay, weight decay 0.01, gradient clip 1.0.
- **Regularization** (this is where the project lives or dies):
  - Dropout 0.2 in encoder; **ticker-dropout 0.15** — randomly mask a subset of tickers per query during training, which both prevents ticker-identity memorization and makes the model robust to universe changes;
  - **Feature-jitter**: small Gaussian noise on the z-scored inputs (σ ≈ 0.05);
  - Early stopping on a **purged validation fold that is never the test fold** — the exact failure that inflated v8 (early-stopping-on-test leakage, see [[06. Context & Logs/Conversations/Conv-2026-06-10-V8-Walkforward-Reanalysis|V8 reanalysis]]) is the #1 thing to not repeat.
- **Seeds**: train 3–5 seeds; report the mean and worst seed, decide on the worst. NN variance at this sample size is large enough to fake an edge in a single lucky seed.

---

## 7. Validation Protocol (non-negotiable)

1. **Same purged, embargoed walk-forward folds** as the v8 reanalysis (9-fold, 2023-08 → present), embargo ≥ 1 day between train and test.
2. **Completed-bar features only**; label window starts strictly after the feature bar closes.
3. **Cost model**: 6 bps per side, applied per side, with the mandatory `median(net − gross) == −cost` per-side audit ([[06. Context & Logs/Daily Logs/2026-06-09|TBM cost-sign lesson]]). Report RAW and NET win-rates separately for long and short.
4. **Baselines that must be beaten, run in the same folds**:
   - B0: v8/v10 XGBoost ranker (ρ ≈ 0.026 raw) — *the incumbent*;
   - B1: XGBoost + the new cross-asset features (the Stage-0 model, §8);
   - B2: **CST with attention ablated** (per-ticker MLP on identical inputs incl. global token) — isolates whether *cross-ticker attention specifically* adds value, vs. just the new features.
5. **Final verdict only via the Validation Gauntlet** (per [[01. Core Architecture/Validation Gauntlet Architecture]]). Note: the Gauntlet's XGBoost-centric harness will need a `predict`-adapter for a PyTorch artifact — scope this before Stage 2. Any metric quoted without a Gauntlet `run_id` stays ⚠️ UNVERIFIED.

### 7.1 Pre-registered kill criteria (decide *before* training)

| Gate | Threshold | Action if failed |
|---|---|---|
| K1 — Raw signal | Avg cross-sectional Spearman ≥ **0.05** across folds (≈2× incumbent) | Kill. Archive to `05. Archives/`. |
| K2 — Non-decay | Latest-2-folds ρ ≥ 60% of full-period average | Kill (a decaying edge is the v8 story again). |
| K3 — Attention value | CST beats B2 (no-attention ablation) by ≥ 0.01 ρ | Kill the transformer, keep the features in XGBoost. |
| K4 — Economic | Top-3 short NET@6bps > 0 with t ≥ 2.0 over the full WF span | No deployment; downgrade to filter/feature research. |
| K5 — Seed robustness | Worst-of-5-seeds still passes K1 | Kill (lucky-seed artifact). |

---

## 8. Staged Execution Plan

**Stage 0 — Falsification test (≈1 day, do this first, no NN code).**
Add cross-asset features to the *existing* XGBoost ranker: lagged Nifty-futures return (1/2/4 bars), sector-index lagged returns, breadth (% of universe above VWAP/open), return-minus-sector, top-3-correlated-peer lagged returns, cross-sectional return dispersion. Re-run the purged WF.
→ If raw ρ does not move meaningfully above 0.026, the cross-name information is **not present at this data granularity**, and the CST is dead on arrival — stop here and redirect effort to the order-flow/microstructure data roadmap. This is the cheapest possible test of the entire hypothesis.

**Stage 1 — Minimal CST (≈3–5 days).** 2 layers, d=64, 1h timeframe, short-side head only (where the real signal is). Single purpose: pass/fail K1–K3.

**Stage 2 — Scale within budget (only if Stage 1 passes).** 15m training (3.6× queries), twin heads, ToD embeddings, 3–5 seeds, hyperparameter sweep ≤ 20 trials (more = multiple-testing rot).

**Stage 3 — Gauntlet.** Build the PyTorch predict-adapter, run the full Gauntlet, obtain `run_id`, write stamped metrics via `scripts/gauntlet/registry.py` only.

**Stage 4 — Deployment economics (only if FILTER_GRADE or better).** Even on success, expected first use is as a **shortlist filter feeding the existing engine** (like the v8/v10 short-side score), not a standalone trade trigger — unless K4 passes decisively.

---

## 9. Risks & Honest Priors

- **Most likely outcome (~60%)**: Stage 0 shows the lead-lag features add little at 15m–1h bar granularity; project stops at zero NN cost. Lead-lag at these horizons is heavily arbitraged on liquid NSE names.
- **Plausible (~30%)**: ρ lifts to 0.03–0.05 — real but below the cost waterline at 6 bps; outcome is a better *filter*, not a trade trigger. Still useful: a stronger shortlist filter compounds with the AI veto layer.
- **Tail (~10%)**: genuine post-cost edge in a pocket (e.g., short-side, specific ToD). History (14:30 pocket, short-side z) says pockets are where it would appear.
- **Structural risks**: ticker-identity overfitting (mitigated: ticker-dropout, embeddings ablation); seed variance faking edges (mitigated: K5); the 5.2K-query budget at 1h (mitigated: train at 15m); regime decay — anything found must be monitored with the same decay lens that demoted v8.
- **What this does not fix**: the 6 bps cost floor and the near-efficiency of bar data. The larger expected payoff remains richer inputs — order-book imbalance, aggressor flow, futures basis — per the Market Psychology roadmap item in [[06. Context & Logs/Current Context]].

---

## 10. Literature Anchors (for orientation, not authority)

- **Feng et al. 2019, "Temporal Relational Ranking for Stock Prediction"** — closest published precedent: temporal encoder per stock + relational/graph attention across stocks + ranking loss.
- **iTransformer (Liu et al. 2023)** — the "invert the attention axis" idea: attend across series, not across time.
- **Gu, Kelly & Xiu 2020** — NN asset pricing baseline; documents the realistic IC scale (0.01–0.05) and that NN gains over trees on tabular features are modest.
- **HATS / graph-attention stock models** — sector/relationship edges help mostly via the *prior*, which the sector embedding here captures more cheaply.

---

## Backlinks
- [[02. Model Suite/Model Performance & Statistics]] — incumbent baselines and raw-signal audit this proposal must beat.
- [[02. Model Suite/Feature Engineering & Normalization]] — reused feature pipeline.
- [[02. Model Suite/Training Data & Regime Requirements]] — 3-year mandate / Row Count Fallacy governing the size budget.
- [[01. Core Architecture/Validation Gauntlet Architecture]] — sole verdict authority.
- [[06. Context & Logs/Conversations/Conv-2026-06-10-Next-Architecture-Advisory|Originating advisory conversation]]
