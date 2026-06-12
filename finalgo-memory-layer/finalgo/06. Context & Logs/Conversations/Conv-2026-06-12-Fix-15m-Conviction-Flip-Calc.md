# 💬 Conversation Context: Fix 15-min Conviction-Flip Exit Calculation

## 📌 Metadata
- **Conversation ID**: (this session, 2026-06-12)
- **Start Date**: 2026-06-12
- **Status**: 🟢 Active
- **Focus Area**: Model Suite (15m scoring) & Trading Strategies (Vanguard exit logic)

## 🎯 Objectives
- [x] Analyse the 15-min conviction-flip exit and find the calculation defect.
- [x] Fix the defects in `orchestrator.py` + `model_inference.py`.
- [ ] User commit approval.

## 💻 Active Code Files Modified
- [model_inference.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/model_inference.py)
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)

## 📝 Compacted Session Log
- **Initial Analysis**: User flagged "something wrong in the calculation" of the 15-min conviction-flip exit. Traced the path: flip exit ([orchestrator.py] flip block) + entry gate both call `_check_15m_percentile`, which ranks `score_15m` from `model_inference.score_15m_universe`.
- **Ruled out (red herring)**: `v3_15min_clean/scaler.pkl` is an *unfitted* `StandardScaler` → `transform()` raises `NotFittedError` → silent fallback to raw features. BUT the training script created it deliberately as a no-op (`StandardScaler(with_mean=False, with_std=False)`, never fitted — `train_ranking_15min_3y.py:382`), so the model was trained on raw features. Raw-at-inference is **correct**; the `[WARN] 15m scaler failed` log is harmless noise. Also confirmed only 4/86 features (the cross-sectional market ones) are zero-filled.
- **Bug #1 (real)**: `_check_15m_percentile` returned `is_in_top=False` for NaN/missing `score_15m`. The 15m fetch is per-ticker under rate limits; on failure a ticker drops to the batch path which yields an **empty 15m frame → NaN score**. A held position with a transient 15m data gap was therefore **force-closed and mislabeled** `Conviction Flip (15m_score=nan)`. Missing data ≠ directional reversal.
- **Bug #2 (real)**: `valid_scores = df["score_15m"].dropna()` — the top-X% quantile was computed over only the names that *won* an individual 15m fetch that cycle (a small, rate-limit-biased subset), not the ~universe. Both the entry gate and the flip exit rode on an unstable mini-cross-section.
- **Bug #3 (methodology)**: `score_15m = (l-mean l) - (s-mean s)` subtracts two uncalibrated `rank:pairwise` margins; the wider-spread leg (short) dominated → persistent short bias (visible in `15m_Conviction_Audit_Report.md`, where L−S is negative for every name regardless of side).
- **Fixes applied**:
  - `model_inference.py`: new static `_combine_long_short(l, s, index)` **z-scores each leg** (unit-variance, zero-std fallback) before differencing; used in 15m/30m/daily scoring. Sign semantics preserved (positive = long-favoured), so research `>0/<0` masks remain valid.
  - `orchestrator.py`: `_check_15m_percentile` now returns a 4-tuple `(is_in_top, score_15m, threshold, is_valid)`; `is_valid=False` when the ticker has no score OR the valid cross-section `< MIN_15M_UNIVERSE (30)`. Flip block **holds** (logs `[CONVICTION-HOLD]`) on indeterminate instead of closing. Entry gate skips with a clear indeterminate log. Both call sites updated.
- **Validation**: both files `ast.parse` clean; `_combine_long_short` unit-checked (legs balanced, sign preserved, zero-std fallback OK).

## ⚠️ Residual (not a plumbing bug)
- The 15m model itself is low-signal (WF Spearman ≈ 0.058, dominated by `IBS`/`Buy_Pressure`); recomputed scores cluster/repeat across tickers in the audit. Even with the plumbing fixed, the flip exit acts on a faint signal — consistent with [[Dual-TF entry/exit research]] showing all 15m overlays sub-cost. Consider whether the flip exit earns its keep at all.

## 🔗 Core Memory Links & Backlinks
- [[02. Model Suite/15m_Conviction_Audit_Report|15m Conviction Audit Report]]
- [[06. Context & Logs/Conversations/Conv-2026-06-11-15m-Model-Audit|15m Model Audit Log]]
