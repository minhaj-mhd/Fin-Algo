# Early-Stage Gatekeeper Feature Definitions

These four features make up the "Early-Stage Gatekeeper" profile of a trade. Before the system runs the heavy, multi-timeframe XGBoost models (15m, 30m, and daily convictions), it evaluates a ticker on these baseline metrics. 

They are used extensively for clustering because they provide a complete environmental "snapshot" of the trade prior to any veto logic being applied.

### 1. `tech_score` (Technical Score)
*   **Definition**: A numerical rating of the ticker's pure technical strength or weakness at the exact moment of evaluation. 
*   **Derivation**: Derived by combining several core technical indicators (such as RSI, MACD crossovers, and distance from moving averages) into a single normalized score. A high positive score indicates strong bullish momentum, while a negative score indicates bearish momentum.

### 2. `nlp_sentiment` (Natural Language Processing Sentiment)
*   **Definition**: A gauge of the current public and financial news sentiment surrounding the ticker. 
*   **Derivation**: Derived by parsing recent news headlines, financial reports, or social media mentions through an NLP (Natural Language Processing) model. Highly positive news pushes the score closer to 1.0, bad news pushes it negative, and no news keeps it at 0.0.

### 3. `tv_sentiment` (TradingView Sentiment Consensus)
*   **Definition**: The aggregate retail technical consensus. 
*   **Derivation**: Pulled directly from the TradingView technical analysis API, which aggregates ~26 different oscillators and moving averages to output a consensus: `STRONG_SELL`, `SELL`, `NEUTRAL`, `BUY`, or `STRONG_BUY`. (For mathematical clustering, these are mapped numerically: -2, -1, 0, 1, 2).

### 4. `one_hour_prob` (1-Hour Baseline Probability)
*   **Definition**: The preliminary machine-learning forecast for the 1-hour timeframe.
*   **Derivation**: A baseline ML model's raw statistical probability that the asset will move in the forecasted direction over the next 1 hour, represented as a percentage (e.g., `35%`).
