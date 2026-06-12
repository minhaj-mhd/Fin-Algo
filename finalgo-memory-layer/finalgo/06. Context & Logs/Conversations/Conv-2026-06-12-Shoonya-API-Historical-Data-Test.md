# 💬 Conversation Context: Shoonya API Historical Data Test

## 📌 Metadata
- **Conversation ID**: b4dda68c-82df-46e2-af4a-46173fa646be
- **Start Date**: 2026-06-12
- **Status**: 🟢 Active
- **Focus Area**: API Exploration

## 🎯 Objectives
- [ ] Test Shoonya API connection.
- [ ] Check how far back we can fetch daily candle data.
- [ ] Check how far back we can fetch hourly candle data.
- [ ] Check how far back we can fetch 15-minute candle data.

## 💻 Active Code Files Modified
- (None yet)

## 📝 Compacted Session Log
- **Initial Analysis**: Bootstrapped, created this session log to isolate context. Now searching the codebase for existing Shoonya API integration or authentication parameters to establish connection.
- **Step 1**: Explored codebase and found `.env` only has `SHOONYA_API_KEY`, `VENDOR_CODE`, `IMEI`.
- **Step 2**: Installed official Python SDK `NorenRestApiPy` and `pyotp`.
- **Step 3**: Created `scripts/research/test_shoonya_historical.py` to test historical limits (15m, 60m, 1D). Script execution paused awaiting user credentials (`SHOONYA_USER_ID`, `SHOONYA_PASSWORD`, `SHOONYA_TOTP_SECRET`) in `.env`.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[04. Data & Code Map/Codebase File Directory]]
