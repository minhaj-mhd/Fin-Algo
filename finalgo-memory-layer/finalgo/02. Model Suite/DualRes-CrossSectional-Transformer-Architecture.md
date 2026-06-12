# 🧠 Dual-Resolution Cross-Sectional Transformer — Architecture

> **Status**: 🟢 In active development (2026-06-12) · ⚠️ UNVERIFIED (no Gauntlet run yet)
> **Conversation**: [[06. Context & Logs/Conversations/Conv-2026-06-12-Sophisticated-Transformer|Build Log]]
> **In-depth flowchart (all inputs/layers/optimizer)**: [[02. Model Suite/DualRes-Transformer-Flowchart]]
> **Code**: [model.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/model.py) ·
> [build_tensor_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/build_tensor_panel.py) ·
> [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py)

## 🎯 Task
For each 1h decision timestamp `t`, predict for **every** ticker whether the **next 1h candle**
closes up or down (`P(next-1h return > 0)`); the sigmoid output doubles as the **confidence**.
Trained on the local **RTX 5050 (8 GB, CUDA, torch 2.11+cu128)**, mixed-precision.

## 🧩 Inputs (per ticker, per decision `t`)
| Stream | Shape | Source |
|---|---|---|
| 1h temporal sequence | `30 × F` | last 30 1h candles (incl. the **14:15 2:15–3:15 context candle**) |
| 15m temporal sequence | `60 × F` | last 60 15m candles, aligned so it closes **with** the 1h bar (15m@T+45) |
| Cross-sectional snapshot | `172 × F` | latest candle of all 172 tickers (the cross-section at `t`) |
| Sector label | `1` | `scripts/sector_map.py` → 16 sectors → learned embedding |
| Daily macro context | `28` | VIX, breadth, Nifty50/500, SP500/NASDAQ/NIKKEI/HSI, USDINR/BRENT/GOLD/DXY/US10Y (PIT, daily) |

`F = 81` vetted, lookahead-free features (`VIEW_A+VIEW_B+VIEW_C` from
[build_feature_views.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/features/build_feature_views.py)),
**per-query cross-sectionally z-scored**. Raw OHLCV and the time-crutch features
(`Hour,DayOfWeek,Is_Open_Hour,Is_Close_Hour,Time_To_Close`) are **excluded as raw inputs**
(proved to overfit a clock in v18/v19) — clock identity is instead injected as a positional/slot
embedding (see below).

## 🏗️ Architecture (`DualResCSTransformer`)
```
 x_1h (N,30,F) ─► Linear→d ─► [+ sinusoidal pos-enc] [+ time-of-day slot emb] ─► Transformer×2 ─► pool ─┐
 x_15m(N,60,F) ─► Linear→d ─► [+ sinusoidal pos-enc] [+ time-of-day slot emb] ─► Transformer×2 ─► pool ─┤
                                                                                                        ├─ concat
 sector_id ───────────────────────────────────────────────────────► Embedding(16,d) ───────────────────┤  → Linear→d (per-ticker token)
                                                                                                        │
 (per-ticker tokens, N of them)  ──►  Cross-Sectional Transformer ×2  (self-attention ACROSS tickers)   │
 macro (28) ─► MLP→d ─► broadcast-add to every ticker token (FiLM-style)  ──────────────────────────────┘
                                   │
                                   ▼
                         LayerNorm→MLP→Linear(d,1)  ─►  logit per ticker  ─►  σ = P(up) = confidence
```
- `d_model=64`, `nhead=4`, temporal layers ×2, cross-sectional layers ×2, GELU, `norm_first=True`, dropout 0.1.
- **Temporal pooling** = last-step embedding + mean-pool of the window.
- **Cross-sectional attention** uses `src_key_padding_mask` to ignore tickers absent at `t`.
- Loss = masked `BCEWithLogitsLoss` on `sign(next-1h return)`; AMP + grad-clip + AdamW + cosine LR.

## ⏱️ Positional / time encoding (answers "are candles identified by time?")
Two complementary signals so candles are placed in time, not just order:
1. **Sinusoidal positional encoding** over sequence position (0…29 for 1h, 0…59 for 15m) — gives the
   model recency/order within each window.
2. **Learned time-of-day slot embedding** — each candle carries its clock slot
   (1h: 6 slots 09:15…14:15; 15m: 25 slots 09:15…15:15) added to its token, so the model can
   distinguish e.g. the open hour from the close hour and detect day boundaries. This is the
   controlled re-introduction of clock info that raw `Hour` caused to overfit in tree models.

## 🔒 Leakage / correctness guarantees
- **Alignment (asserted in builder)**: 1h@T (left-labelled) ⟺ 15m@(T+45m); 15m window close-time == 1h close-time.
- **No overnight leak**: forward returns are **session-masked** (NaN at each day's last bar). The
  **14:15 candle is kept as INPUT CONTEXT only** (label = NaN → never a decision point; it is already
  the label target of the 13:15 decision). Built from the four 15m bars [14:15,14:30,14:45,15:00].
- **Single source**: both timeframes resampled from the raw 15m cache (origin='start_day', offset='15min'),
  reproducing [rebuild_aligned_datasets.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/rebuild_aligned_datasets.py).
- Label base rate (up) ≈ **47.9%** → "short" is the easier RAW side; honest eval must compare RAW vs
  NET win-rate **per side** and check `median(net−gross)==−cost` (see [[feedback_validate_cost_accounting]]).

## ⚠️ Prior-evidence context (must be beaten, not assumed away)
This recombines approaches the repo independently killed: cross-sectional/lead-lag at 1h
([[project_cst_stage0_killed]]), directional next-candle classification ([[project_v18_v19_directional_deadend]]),
and 1h price/volume having no post-cost edge ([[project_tbm_1h_ensemble_results]], [[project_v8_1h_walkforward_demoted]]).
The macro/VIX/sector fusion already exists at daily granularity and is Gauntlet **DEAD/DEAD**
([[project_validation_gauntlet]]). **Novelty** = first end-to-end temporal+cross-sectional *sequence*
transformer (prior CST test only bolted 11 lead-lag features onto the tree ranker) + intraday×macro fusion.
Verdict authority rests **only** with the Validation Gauntlet; eval scripts here are exploratory.

## 💰 Costs
Gauntlet binding round-trip cost = **10 bps** (also reports 6 bps) — `scripts/gauntlet/contracts.py`.
Net-of-cost Top-1/Top-3 per side is the tradeable metric.
