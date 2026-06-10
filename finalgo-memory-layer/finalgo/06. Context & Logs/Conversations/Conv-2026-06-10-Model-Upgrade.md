# 💬 Conversation Context: Gemini 3 Model Upgrade

## 📌 Metadata
- **Conversation ID**: 11bb4fe1-a212-4d7f-818d-f8e6633f8f92
- **Start Date**: 2026-06-10
- **Status**: 🔴 Concluded
- **Focus Area**: AI Veto & Gemini Audit

## 🎯 Objectives
- [x] Migrate system from legacy 2.5 models to Gemini 3.5 Flash and Gemini 3 Flash Preview.
- [x] Handle Google Search Grounding quota restrictions on the Preview model.

## 💻 Active Code Files Modified
- [scripts/vanguard/config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py)
- [scripts/vanguard/ai_veto.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/ai_veto.py)
- [scripts/ticker_intelligence.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/ticker_intelligence.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The engine was hardcoded to `gemini-2.5-flash` which was experiencing severe free-tier API rate limits (15 RPM) and blocking live analysis.
- **Verification**: Executed a background test of `gemini-3.5-flash` and `gemini-3-flash-preview`. Both models succeeded instantly without quota issues for raw text generation.
- **Grounding Limitation Discovered**: Discovered that enabling Google Search Grounding on `gemini-3-flash-preview` results in an immediate, unrecoverable `429 RESOURCE_EXHAUSTED` error without a retry delay on the user's free tier.
- **Resolution**: Refactored `ai_veto.py` and `config.py` to split the model tiers:
  - **Stage 1 (Technical Veto - No Tools)**: Uses `gemini-3-flash-preview` ➔ `gemini-3.1-flash-lite` ➔ `gemini-2.5-flash-lite`.
  - **Stage 2 (News Audit - Search Grounding)**: Restricted exclusively to `gemini-2.5-flash-lite` to guarantee Search Grounding availability and bypass heavy free-tier rate limits.
  - **Lite Model Parser Hotfix**: Added a highly robust plain-text `[BLOCK-RECOVERY]` parser to `ai_veto.py` and converted the Stage 2 prompt to request rigid `[KEY] Value` plain-text instead of JSON. This entirely prevents `gemini-2.5-flash-lite` from failing due to complex JSON formatting requirements.
- **Rotator Enhancements**: 
  - **Hard Quota Fast-Fail**: Fixed a bug where a hard quota hit (429 without a retry delay) would default to a 60s cooldown. The rotator now instantly breaks the retry loop on a hard quota hit.
  - **Rate Limit Sleep Removal**: Removed the blocking `time.sleep` when all keys hit their rate limits (e.g., 15 RPM). Instead of freezing the engine for up to 60 seconds to ride out the limits, it now fails fast to instantly trigger a seamless fallback to the next model tier (e.g., from `2.5-flash` to `2.5-flash-lite`).
  - Added a `random.uniform(3.0, 5.0)` second jitter delay between failed key rotation attempts to prevent hammering the API.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[04. Data & Code Map/AI Veto & Gemini Audit]]
