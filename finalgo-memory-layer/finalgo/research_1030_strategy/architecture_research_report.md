# Intraday Stock Trading: AI Architecture Research Report

## 1. Problem Formulation
**Objective**: Predict the intraday trend, returns, or distribution for the rest of the day based on:
1. **Sequential Data**: The first 1.25 hours of trading (high-frequency price action, volume, order flow).
2. **Tabular/Structured Data**: Overnight global data (Asian/European market indices, pre-market futures, macroeconomic releases).

This represents a classic **Multimodal Time-Series Forecasting** problem, blending high-frequency noisy signals (price action) with lower-frequency structural signals (overnight sentiment/indices).

---

## 2. Transformers for Time-Series
Transformers rely on self-attention mechanisms, making them powerful for finding complex, long-range correlations across sequences.

*   **Multihead Attention & Temporal Attention**: Allows the model to look at various parts of the sequence simultaneously (e.g., comparing the first 5 minutes directly with the 60th minute) to identify non-linear relationships driving market behavior.
*   **Time-Series Specific Transformers**:
    *   **Informer**: Utilizes "ProbSparse" attention, which reduces the massive O(N^2) computational complexity of standard Transformers to O(N log N). Excellent for longer sequences like 1-minute or tick-level intraday data.
    *   **Autoformer**: Features a built-in decomposition block that splits data into "trend" and "seasonal" components natively. This is highly useful for separating the underlying daily trend from intraday volatility and mean-reverting noise.
    *   **TimeGPT**: Large pre-trained foundation models that can be fine-tuned. While powerful, they often require heavy domain-specific adaptation to account for the unique microstructure of intraday markets.

**Pros for Financial Time-Series**:
- Can capture extremely complex non-linear relationships.
- Excellent at discovering global dependencies in long sequence data.

**Cons for Financial Time-Series**:
- **Overfitting**: Highly prone to "memorizing" market noise rather than extracting true signal due to their immense parameter counts.
- **Compute Cost**: Slow to train and latency-heavy during inference, which can be detrimental in real-time trading environments.
- Often outperformed by simpler, well-regularized models on financial datasets due to the low signal-to-noise ratio.

---

## 3. Gradient Boosting (XGBoost, LightGBM, CatBoost)
Tree-based ensemble models remain the industry "workhorses" for tabular data and engineered features.

**Pros for Financial Time-Series**:
- **Efficiency & Speed**: Extremely fast to train and infer (low latency), perfect for real-time intraday trading.
- **Robustness**: Inherently resilient to noisy data and significantly less prone to catastrophic overfitting compared to deep neural networks.
- **Feature Importance**: High interpretability allows quants to understand which global indices or price indicators are driving the predictions.

**Cons for Financial Time-Series**:
- **Sequential Ignorance**: They do not natively understand "time". Sequential data must be heavily feature-engineered (e.g., rolling averages, RSI, MACD, volatility lags) into tabular format, which may miss subtle temporal dynamics.

---

## 4. Other Architectures: LSTMs, CNNs, and State Space Models (Mamba)

*   **LSTMs/GRUs**: Traditional deep learning for time series. Good at remembering sequential states but suffer from vanishing gradients over very long sequences (e.g., tick data) and take a long time to train sequentially.
*   **1D CNNs**: Excellent for extracting "spatial" local patterns (like specific candlestick formations or micro-trends) across time windows. Often faster and less prone to overfitting than LSTMs.
*   **State Space Models (e.g., Mamba)**:
    *   **The Challenger**: Mamba offers the expressive power of Transformers but with **linear O(N) scaling** instead of quadratic.
    *   **Why it matters**: It can process extremely long contexts (like tens of thousands of ticks from the first 1.25 hours) lightning-fast, providing the low-latency inference critical for intraday trading.
    *   **Pros**: Fast inference, handles long sequences efficiently without the memory bottleneck of attention.
    *   **Cons**: Emerging technology, lacks the extensive ecosystem, tooling, and battle-testing of Transformers or XGBoost.

---

## 5. Best Architecture for Combining Tabular and Sequential Data

To effectively combine **structured tabular data** (overnight global indices) with **sequential price action** (the first 1.25 hours), a **Hybrid "Late Fusion" Architecture** is the most optimal approach. Early fusion (concatenating raw inputs) often fails because the models struggle to balance the low-frequency tabular data with high-frequency sequence data.

### The Recommended Architecture: Two-Tower Hybrid (Deep Sequence + Gradient Boosting)

1. **Sequence Tower (Mamba or Autoformer)**:
   - Feeds exclusively on the raw sequential price action and volume data of the first 1.25 hours.
   - Extracts a dense, low-dimensional "Temporal Embedding" vector representing the intraday market state.
   - *Recommendation*: Use **Mamba** for tick-level/second-level data due to its O(N) speed, or **Autoformer** if using 1-minute/5-minute bars to leverage its trend decomposition.

2. **Tabular Integration (LightGBM/XGBoost Meta-Learner)**:
   - Instead of using a standard neural network for the final prediction, feed the extracted Temporal Embedding (from the Sequence Tower) **as a feature** alongside your raw tabular overnight global indices into a **LightGBM or XGBoost model**.

### Why this is the best choice:
- **Plays to strengths**: Lets deep learning handle the complex sequential feature extraction of price action, while gradient boosting handles the non-linear interactions of the static overnight/tabular features.
- **Robustness & Regularization**: Gradient boosting acts as a powerful regularizer at the end of the pipeline, preventing the deep learning sequence model from over-extrapolating on market noise.
- **Handling Dimensionality**: Trees handle the structural differences between dense neural embeddings and discrete/continuous tabular data beautifully without requiring complex scaling or normalization.

## 6. Conclusion
While Time-Series Transformers like Autoformer and Informer are conceptually brilliant, their computational overhead and propensity for overfitting make them tricky for raw intraday prediction. **State Space Models like Mamba** represent a highly promising future for processing high-frequency intraday price action due to their linear efficiency. 

However, for the specific task of combining overnight global data with 1.25 hours of trading, the most robust, battle-tested approach is a **Late Fusion Hybrid Model**. Leveraging a deep learning sequence model (Mamba or 1D-CNN) to extract a temporal state embedding, and fusing that embedding with overnight indices via an **XGBoost/LightGBM meta-learner**, provides the optimal balance of temporal awareness, robustness to noise, and low latency.
