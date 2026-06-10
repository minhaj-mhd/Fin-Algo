# Advanced Tree Models Roadmap

Having pushed standard XGBoost models (using `reg:squarederror`, `binary:logistic`, and `rank:ndcg`) to their limits by tuning structural hyperparameters (e.g., restricting `max_depth` to regularize noise), we have identified five advanced techniques within the tree-based algorithmic family. These avenues represent the next frontier for combating the high noise-to-signal ratio inherent in intraday financial data and structural fee erosion.

---

## 1. Alternative Boosting Engines (CatBoost & LightGBM)
While we have exclusively used XGBoost, other modern gradient boosting frameworks offer specific architectural advantages:
* **CatBoost**: Employs "Oblivious" (symmetric) trees. These are mathematically proven to be far more resistant to overfitting on extremely noisy datasets compared to XGBoost's standard asymmetric trees. If we want to safely increase trade volume by lowering conviction thresholds, CatBoost's structural regularization might prevent the performance collapse we observed with XGBoost.
* **LightGBM**: Utilizes "leaf-wise" tree growth rather than XGBoost's depth-wise growth. LightGBM excels at isolating very narrow, deep, highly profitable asymmetric pockets of alpha that XGBoost's uniform depth expansion might gloss over.

## 2. Custom Asymmetric Objective Functions
Our current XGBoost classifiers use standard `logloss`, treating a missed trade (false negative) and a losing trade (false positive) with equal penalty.
* **The Solution**: Develop a custom Python objective function for XGBoost that explicitly bakes in the 10 bps fee structure. By mathematically penalizing false positives (failed breakouts that incur fees) 5x harder than false negatives (missed opportunities), we force the tree splits to be hyper-conservative and cost-aware natively during the gradient descent process.

## 3. Monotonic Constraints
Tree models are susceptible to learning non-sensical relationships due to market noise (e.g., learning to short when RSI is 80, but go long when RSI is 85 due to a random historical outlier).
* **The Solution**: Apply Monotonic Constraints to XGBoost, forcing the model to adhere to foundational economic logic. We can mathematically restrict the tree so that, for example, as volatility increases, the model's confidence in Long breakouts is *forced* to decrease monotonically. This acts as the ultimate structural regularizer.

## 4. DART (Dropout Additive Regression Trees)
Instead of standard Gradient Boosting where each sequential tree fiercely attempts to correct the residuals of the previous ones, DART randomly "drops out" existing trees during training (analogous to neural network dropout).
* **The Advantage**: In finance, the "residuals" left for the later trees to fix are almost always pure noise. DART explicitly prevents the later trees from overfitting to that noise, yielding a much smoother, more robust model that survives out-of-sample walk-forward validation better.

## 5. Random Forests (Bagging vs. Boosting)
Gradient Boosting (XGBoost) sequentially hunts down errors. Random Forests, conversely, build hundreds of shallow, independent trees simultaneously and average their predictions. 
* **The Advantage**: Because financial data exhibits an exceptionally low signal-to-noise ratio, the independent averaging process of a Random Forest (bagging) can sometimes outperform the aggressive, sequential error-hunting of XGBoost (boosting), providing a more stable edge.
