# 🗃️ Model Registry & File Structures

The Vanguard engine maintains a highly structured, registry-controlled model management system under the `models/` directory. Centralized in `scripts/model_registry.py`, this system validates model files, registers new training runs, and manages dynamic switching of the active system configuration.

---

## 📂 Models Directory Layout

All trained machine learning files, feature scalers, and registries are organized systematically under `models/`:

```text
models/
├── registry.json                    # Central database of all registered model versions
├── model_metadata.json              # Legacy active model feature list
├── scaler.pkl                       # Pickled Feature Scaler (StandardScaler)
├── daily_xgb/                       # Daily macro XGBoost gatekeeper model files
│   ├── xgb_long_model.json
│   └── xgb_short_model.json
├── daily_transformer/               # Daily macro PyTorch Transformer (Retired/Inactive)
├── v8_upstox_3y/                    # V8 core trend model (trained on 3 years of Upstox hourly bars)
│   ├── xgb_long_model.json
│   ├── xgb_short_model.json
│   └── metadata.json
└── v1_15min/                        # 15-Minute stand-alone ranking model (Archive)
```

---

## 📄 Active Registry Schema (`models/registry.json`)

The active model registry uses a standardized schema that groups models by name and details their technical paths, evaluation scores, and metadata.

```json
{
  "active_model": "v8_upstox_3y",
  "models": {
    "v8_upstox_3y": {
      "description": "V8: Trained on 3 years of hourly Upstox data with microstructure features (IBS, Buy_Pressure, etc.).",
      "long_model": "models/v8_upstox_3y/xgb_long_model.json",
      "short_model": "models/v8_upstox_3y/xgb_short_model.json",
      "scaler": "models/scaler.pkl",
      "meta": "models/v8_upstox_3y/metadata.json",
      "type": "ranker",
      "long_test_spearman": 0.0461,
      "short_test_spearman": 0.049,
      "trained_at": "2026-05-28T14:04:01",
      "notes": "Added microstructure features. Huge performance boost."
    },
    "v4_regime_aware": {
      "description": "V4 Regime-Aware + Multi-TF model. Optimized for stable alpha.",
      "long_model": "models/v4_regime_aware/xgb_long_model.json",
      "short_model": "models/v4_regime_aware/xgb_short_model.json",
      "scaler": "",
      "meta": "models/v4_regime_aware/metadata.json",
      "type": "ranker",
      "long_test_spearman": 0.082,
      "short_test_spearman": 0.085,
      "trained_at": "2026-05-17T00:33:08",
      "notes": "Regime-conditioned ranker. Scale-invariant."
    }
  }
}
```

### Key Schema Fields
*   `active_model`: The key of the model loaded by the engine at startup.
*   `long_model` / `short_model`: Path to the serialized XGBoost JSON boosters.
*   `scaler`: Path to the pickled scaler. An empty string (`""`) signifies a **scale-free** setup.
*   `meta`: Path to the JSON metadata file detailing required feature column names.
*   `type`: Classification type (`ranker`, `ranker_15min`, `ranker_30min`).
*   `long_test_spearman` / `short_test_spearman`: Performance Rho on out-of-sample backtests.

---

## 🧬 Feature Normalization & Scale-Invariance

### 1. Scaler Standardisation (`models/scaler.pkl`)
For models that require standardized inputs (like V2, V3, and V6), the feature generator passes raw calculations through a `StandardScaler`:
```python
if resolved_scaler and os.path.exists(resolved_scaler):
    self.scaler = pickle.load(open(resolved_scaler, "rb"))
```
Standardizing variables ensures that dynamic ranges (e.g. stock price in INR vs. RSI between 0-100) do not distort inference during the ranker pass.

### 2. Scale-Invariant Models
*   **Concept**: Since **XGBoost** is a decision tree ensemble, its splits are based on relative feature inequalities (`feature > threshold`) rather than distance. Trees are inherently **scale-invariant**.
*   **Scaler-Free Execution**: Several advanced model configurations (like V4 and V8) leave the `scaler` parameter blank in the registry. The engine detects this automatically and passes features straight to the model booster without scaler transformation, completely avoiding scale distortion risks:
    ```python
    self.scaler = None
    log("No scaler configured (scale-invariant XGBoost)")
    ```

---

## 🛠️ Model Registry CLI Commands

The registry exposes a simple command-line interface to manage active configurations:
*   **List all models**:
    ```powershell
    python scripts/model_registry.py --list
    ```
*   **Show active model info**:
    ```powershell
    python scripts/model_registry.py --active
    ```
*   **Switch the active model Safely**:
    The script programmatically validates that all JSON boosters, metadata, and scaler files exist in the file system before switching the registry pointer:
    ```powershell
    python scripts/model_registry.py --switch v8_upstox_3y
    ```
*   **Dry-run verification**:
    ```powershell
    python scripts/model_registry.py --switch v8_upstox_3y --dry-run
    ```

---

## 👁️ Key Related Notes
*   Review our detailed training metrics, walk-forward folds, and hyperparameter tables: [[02 — Models/_Shared/Model Performance & Statistics|Model Performance & Statistics]].
*   Review our feature utils file equations: [[01 — Architecture/Data & Code/Codebase File Directory|Codebase File Directory]].
*   See the multi-timeframe context inputs: [[02 — Models/_Shared/Multi-Timeframe Models|Multi-Timeframe Models]].
*   See how registry models are loaded at startup: [[01 — Architecture/Global System Architecture|Global System Architecture]].
