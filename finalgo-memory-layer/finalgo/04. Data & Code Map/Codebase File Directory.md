# 📂 Codebase File Directory

The Vanguard system consists of a structured suite of Python scripts, backtesters, database models, dashboards, and configurations. Below is a comprehensive index mapping files in our project workspace by their purpose.

---

## 🏛️ 1. Core Executables & Live Engines
*   **[`scripts/vanguard_signal_engine.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard_signal_engine.py)**: The central trading execution engine. Handles live data fetching, daily macro scans, hourly core model ranking, dynamic S/R pivoted calculations, AI (Gemini) auditing, shadow tracking loops, entry confirmations, dynamic ATR brackets, and Upstox API coordination.
*   **[`scripts/vanguard_dashboard.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard_dashboard.py)**: Web-based real-time dashboard UI for trade monitoring, portfolio metrics, equity curves, and AI Veto performance auditing.
*   **[`run_vanguard_system.bat`](file:///c:/Users/loq/Desktop/Trading/finalgo/run_vanguard_system.bat)**: Admin shell script to boot yfinance daily scanners and activate the Vanguard live system.
*   **[`run_upstox_debug.bat`](file:///c:/Users/loq/Desktop/Trading/finalgo/run_upstox_debug.bat)**: Batch file to test active Upstox access tokens and verify connectivity.
*   **[`run_error_monitor.bat`](file:///c:/Users/loq/Desktop/Trading/finalgo/run_error_monitor.bat)**: Shell script to stream live engine error logs.

---

## 📈 2. Backtesters & Strategy Simulators
*   **[`scripts/strategy_filters.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_filters.py)**: Encapsulates structural rule filters for Strategy 1 through 50 (e.g. ORB volatility bands, stochastic zones, moving average cross-confluences).
*   **[`scripts/strategy_10x_backtest.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_10x_backtest.py)**: Backtests the 10 core strategies (e.g., Strategy 8 ORB, Strategy 10 Quad-TF) across historical NSE data with strict 0.06% round-trip friction.
*   **[`scripts/strategy_25x_backtest.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_25x_backtest.py)**: Advanced simulator covering 25 refined mean-reversion and trend strategies.
*   **[`scripts/strategy_35x_backtest.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_35x_backtest.py)**: Deep simulator exploring 35 complex timeframe confluence setups.
*   **[`scripts/backtest_multi_tf_v2.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtest_multi_tf_v2.py)**: Multi-timeframe execution backtest sweep supporting trailing stops and profit margins.

---

## ⚙️ 3. Training & Data Collection Pipelines
*   **[`scripts/feature_utils.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/feature_utils.py)**: The central numeric indicator calculator. Computes SMA, EMA, WMA, HMA, RSI, ROC, CCI, Stochastic, Bollinger Bands, OBV, CMF, and custom daily/15M/30M feature sets. Includes microstructural calculation modules (IBS, Buy Pressure).
*   **[`scripts/train_ranking.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/train_ranking.py)**: Core trainer for standard XGBoost relative ranking models.
*   **[`scripts/collect_upstox_3y.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collect_upstox_3y.py)**: Data gatherer connecting to Upstox API to cache 3 years of 1-Hour candles for training.
*   **[`scripts/collect_upstox_daily_5y.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collect_upstox_daily_5y.py)**: Data collector for 5 years of daily NSE stock histories.

---

## 💾 4. Database, Broker & Infrastructure Map
*   **[`scripts/database_manager.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/database_manager.py)**: Central SQLite manager. Handles migrations, trade updates, monthly/weekly/daily performance queries, and veto statistics.
*   **[`scripts/upstox_broker.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/upstox_broker.py)**: Sandbox execution broker client encapsulating order placement, portfolio fetches, and transaction fee math.
*   **[`scripts/upstox_websocket.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/upstox_websocket.py)**: Full-duplex WebSocket feed client to stream real-time price updates into the engine's price cache.
*   **[`scripts/model_registry.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/model_registry.py)**: Registry manager to switch active models safely and print registry status configurations.

---

## 🗄️ 5. Legacy Archive & Isolated Assets
*   **[`legacy_archive/`](file:///c:/Users/loq/Desktop/Trading/finalgo/legacy_archive)**: Central repository for isolated stale assets. Holds retired PyTorch daily temporal transformers, obsolete diagnostic sweeping scripts, V1 inverted XGBoost documentation, and pre-V2.3 research notes. Managed in a non-destructive phased manner.

---

## 👁️ Key Related Notes
*   See how model files are loaded: [[02. Model Suite/Model Registry & File Structures|Model Registry & File Structures]].
*   See our core DB schemas: [[04. Data & Code Map/Database Architecture|Database Architecture]].
*   Review our live risk loops: [[04. Data & Code Map/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]].
