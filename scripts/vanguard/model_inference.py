import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scripts.vanguard import config
from scripts.terminal_utils import log

class ModelManager:
    def __init__(self):
        self.bst_long = None
        self.bst_short = None
        self.scaler = None
        self.feature_cols = []
        self.active_model_name = config.DEFAULT_MODEL_NAME

        self.daily_xgb_long = None
        self.daily_xgb_short = None
        self.daily_feature_cols = []

    def load_active_models(self, model_path_fallback=None, scaler_path_fallback=None, meta_path_fallback=None):
        """Loads live trading XGBoost models via registry or legacy fallbacks."""
        try:
            from scripts.model_registry import ModelRegistry
            registry = ModelRegistry()
            active = registry.get_active_model()
            long_model_path = active["long_model"]
            short_model_path = active["short_model"]
            resolved_meta = active["meta"]
            resolved_scaler = active["scaler"]
            self.active_model_name = active["name"]
            log(f"[REGISTRY] Active model: {active['name']}")
        except Exception as reg_err:
            log(f"[WARN] Registry unavailable ({reg_err}). Using legacy paths.")
            if model_path_fallback:
                model_dir = os.path.dirname(model_path_fallback)
                long_model_path = os.path.join(model_dir, config.DEFAULT_LONG_MODEL_NAME)
                short_model_path = os.path.join(model_dir, config.DEFAULT_SHORT_MODEL_NAME)
                resolved_meta = meta_path_fallback
                resolved_scaler = scaler_path_fallback
            else:
                model_dir = config.MODEL_REGISTRY_FALLBACK_DIR
                long_model_path = os.path.join(model_dir, config.DEFAULT_LONG_MODEL_NAME)
                short_model_path = os.path.join(model_dir, config.DEFAULT_SHORT_MODEL_NAME)
                resolved_meta = os.path.join(model_dir, "metadata.json")
                resolved_scaler = os.path.join(model_dir, "scaler.pkl")
            self.active_model_name = "legacy_fallback"

        try:
            self.bst_long = xgb.Booster()
            self.bst_long.load_model(long_model_path)
            self.bst_long.set_param({'device': 'cuda'})

            self.bst_short = xgb.Booster()
            self.bst_short.load_model(short_model_path)
            self.bst_short.set_param({'device': 'cuda'})

            if resolved_scaler and os.path.exists(resolved_scaler):
                with open(resolved_scaler, "rb") as f:
                    self.scaler = pickle.load(f)
                log(f"[INFO] Scaler loaded from {resolved_scaler}")
            else:
                self.scaler = None
                log(f"[INFO] No scaler configured (scale-invariant XGBoost)")

            if os.path.exists(resolved_meta):
                with open(resolved_meta, "r") as f:
                    self.feature_cols = json.load(f)["features"]
                log(f"[OK] ML Models: {len(self.feature_cols)} features | LOADED ({self.active_model_name})")
            else:
                raise FileNotFoundError(f"Metadata file not found: {resolved_meta}")

            # Validation Gauntlet Verification Guard
            try:
                from scripts.gauntlet.registry import verify_model_stamp
                model_dir = os.path.dirname(long_model_path)
                verification = verify_model_stamp(model_dir)
                
                if not verification["valid"]:
                    msg = f"[GAUNTLET] WARNING: Model '{self.active_model_name}' failed stamp verification: {verification['reason']}"
                    log(msg)
                    if config.GAUNTLET_ENFORCEMENT == "enforce":
                        log("[GAUNTLET] CRITICAL: Hard enforcement enabled. Refusing to load model. Exiting.")
                        sys.exit(1)
                else:
                    verdict = verification["verdict"]
                    log(f"[GAUNTLET] OK: Model verified. Verdicts: Long={verdict['long']}, Short={verdict['short']}")
                    for side in ["long", "short"]:
                        grade = verdict[side]
                        if grade == "DEAD":
                            log(f"[GAUNTLET] WARNING: {side.upper()} side is DEAD according to gauntlet stamp.")
                            if config.GAUNTLET_ENFORCEMENT == "enforce":
                                log(f"[GAUNTLET] CRITICAL: {side.upper()} side is DEAD. Enforcement enabled. Exiting.")
                                sys.exit(1)
            except Exception as guard_err:
                log(f"[GAUNTLET] WARNING: Failed to execute gauntlet guard verification: {guard_err}")
                if config.GAUNTLET_ENFORCEMENT == "enforce":
                    log("[GAUNTLET] CRITICAL: Enforcement enabled but guard crashed. Exiting.")
                    sys.exit(1)

        except Exception as e:
            log(f"[ERROR] Critical ML Model Load Error: {e}")
            sys.exit(1)

    def load_daily_gatekeepers(self):
        """Loads XGBoost daily trend gatekeeper models."""
        try:
            log("[INFO] Loading Daily Macro Trend Gatekeeper Models...")
            self.daily_xgb_long = xgb.Booster()
            self.daily_xgb_long.load_model(config.DAILY_MACRO_LONG_PATH)
            self.daily_xgb_long.set_param({'device': 'cuda'})

            self.daily_xgb_short = xgb.Booster()
            self.daily_xgb_short.load_model(config.DAILY_MACRO_SHORT_PATH)
            self.daily_xgb_short.set_param({'device': 'cuda'})

            if os.path.exists(config.DAILY_MACRO_META_PATH):
                with open(config.DAILY_MACRO_META_PATH, "r") as f:
                    daily_meta = json.load(f)
                self.daily_feature_cols = daily_meta["features"]
                log("[OK] Daily Macro Gatekeepers loaded successfully.")
            else:
                raise FileNotFoundError(f"Daily metadata file not found: {config.DAILY_MACRO_META_PATH}")

        except Exception as daily_load_err:
            log(f"[ERROR] Failed to load Daily Macro Gatekeepers: {daily_load_err}")
            sys.exit(1)

    def load_multi_tf_models(self):
        """Loads additional timeframe models for dashboard tracking."""
        try:
            log("[INFO] Loading Multi-Timeframe Dashboard Models...")
            
            # 15m
            self.tf_15m_long = xgb.Booster()
            self.tf_15m_long.load_model("models/v3_15min_clean/xgb_long_model.json")
            self.tf_15m_long.set_param({'device': 'cuda'})
            self.tf_15m_short = xgb.Booster()
            self.tf_15m_short.load_model("models/v3_15min_clean/xgb_short_model.json")
            self.tf_15m_short.set_param({'device': 'cuda'})
            with open("models/v3_15min_clean/metadata.json", "r") as f:
                self.tf_15m_features = json.load(f)["features"]
            if os.path.exists("models/v3_15min_clean/scaler.pkl"):
                with open("models/v3_15min_clean/scaler.pkl", "rb") as f:
                    self.tf_15m_scaler = pickle.load(f)
            else:
                self.tf_15m_scaler = None

            # 30m
            self.tf_30m_long = xgb.Booster()
            self.tf_30m_long.load_model("models/v1_30min/xgb_long_model.json")
            self.tf_30m_long.set_param({'device': 'cuda'})
            self.tf_30m_short = xgb.Booster()
            self.tf_30m_short.load_model("models/v1_30min/xgb_short_model.json")
            self.tf_30m_short.set_param({'device': 'cuda'})
            with open("models/v1_30min/metadata.json", "r") as f:
                self.tf_30m_features = json.load(f)["features"]
            if os.path.exists("models/v1_30min/scaler.pkl"):
                with open("models/v1_30min/scaler.pkl", "rb") as f:
                    self.tf_30m_scaler = pickle.load(f)
            else:
                self.tf_30m_scaler = None

            # Daily (separate from macro gatekeeper)
            self.tf_daily_long = xgb.Booster()
            self.tf_daily_long.load_model("models/daily_xgb_v2/xgb_long_model.json")
            self.tf_daily_long.set_param({'device': 'cuda'})
            self.tf_daily_short = xgb.Booster()
            self.tf_daily_short.load_model("models/daily_xgb_v2/xgb_short_model.json")
            self.tf_daily_short.set_param({'device': 'cuda'})
            with open("models/daily_xgb_v2/metadata.json", "r") as f:
                self.tf_daily_features = json.load(f)["features"]
            if os.path.exists("models/daily_xgb_v2/scaler.pkl"):
                with open("models/daily_xgb_v2/scaler.pkl", "rb") as f:
                    self.tf_daily_scaler = pickle.load(f)
            else:
                self.tf_daily_scaler = None

            log("[OK] Multi-Timeframe Dashboard models loaded successfully.")
        except Exception as e:
            log(f"[WARN] Failed to load Multi-Timeframe Dashboard models: {e}")

    def score_universe(self, scores_df, ticker_metadata):
        """Applies Z-scoring and runs inference on features DataFrame for the ticker universe."""
        if scores_df.empty:
            return pd.DataFrame()

        # Z-Scoring has been completely removed because the model was trained on raw features.

        try:
            missing = [c for c in self.feature_cols if c not in scores_df.columns]
            if missing:
                log(f"[ERROR] Missing feature columns: {missing}")
                return pd.DataFrame()

            X = scores_df[self.feature_cols].values
            X_clean = np.nan_to_num(X)

            scaler_is_fitted = (
                self.scaler is not None
                and hasattr(self.scaler, 'scale_')
                and self.scaler.scale_ is not None
            )
            if scaler_is_fitted:
                X_final = self.scaler.transform(X_clean)
                log(f"[INFO] Scaler applied ({self.active_model_name})")
            else:
                X_final = X_clean

            dmatrix = xgb.DMatrix(X_final, feature_names=self.feature_cols)
            scores_df = scores_df.copy()

            long_score = self.bst_long.predict(dmatrix)
            short_score = self.bst_short.predict(dmatrix)
            
            l_centered = long_score - np.mean(long_score)
            s_centered = short_score - np.mean(short_score)
            
            long_conviction = l_centered - s_centered
            short_conviction = s_centered - l_centered
            
            scores_df = scores_df.assign(
                long_score=long_score,
                short_score=short_score,
                Long_Conviction=long_conviction,
                Short_Conviction=short_conviction,
                Long_Rank=pd.Series(long_conviction).rank(ascending=False).values,
                Short_Rank=pd.Series(short_conviction).rank(ascending=False).values
            ).copy()

            return scores_df

        except Exception as e:
            log(f"[ERROR] Prediction Error: {e}")
            return pd.DataFrame()

    def score_15m_universe(self, df_15m):
        """Scores the 15-minute universe dataframe."""
        if not hasattr(self, 'tf_15m_long') or not self.tf_15m_long or df_15m.empty:
            return pd.Series(dtype=float)
        try:
            missing = [c for c in self.tf_15m_features if c not in df_15m.columns]
            if missing:
                missing_df = pd.DataFrame(0.0, index=df_15m.index, columns=missing)
                df_15m = pd.concat([df_15m, missing_df], axis=1)
            
            X_clean = np.nan_to_num(df_15m[self.tf_15m_features].values)
            
            if hasattr(self, 'tf_15m_scaler') and self.tf_15m_scaler is not None:
                try:
                    X_final = self.tf_15m_scaler.transform(X_clean)
                except Exception as e:
                    log(f"[WARN] 15m scaler failed: {e}")
                    X_final = X_clean
            else:
                X_final = X_clean
                
            dmatrix = xgb.DMatrix(X_final, feature_names=self.tf_15m_features)
            l = self.tf_15m_long.predict(dmatrix)
            s = self.tf_15m_short.predict(dmatrix)
            return pd.Series((l - np.mean(l)) - (s - np.mean(s)), index=df_15m.index)
        except Exception as e:
            log(f"[WARN] 15m scoring failed: {e}")
            return pd.Series(dtype=float, index=df_15m.index)

    def score_30m_universe(self, df_30m):
        """Scores the 30-minute universe dataframe."""
        if not hasattr(self, 'tf_30m_long') or not self.tf_30m_long or df_30m.empty:
            return pd.Series(dtype=float)
        try:
            missing = [c for c in self.tf_30m_features if c not in df_30m.columns]
            if missing:
                missing_df = pd.DataFrame(0.0, index=df_30m.index, columns=missing)
                df_30m = pd.concat([df_30m, missing_df], axis=1)
            
            X_clean = np.nan_to_num(df_30m[self.tf_30m_features].values)
            
            if hasattr(self, 'tf_30m_scaler') and self.tf_30m_scaler is not None:
                try:
                    X_final = self.tf_30m_scaler.transform(X_clean)
                except Exception as e:
                    log(f"[WARN] 30m scaler failed: {e}")
                    X_final = X_clean
            else:
                X_final = X_clean
                
            dmatrix = xgb.DMatrix(X_final, feature_names=self.tf_30m_features)
            l = self.tf_30m_long.predict(dmatrix)
            s = self.tf_30m_short.predict(dmatrix)
            return pd.Series((l - np.mean(l)) - (s - np.mean(s)), index=df_30m.index)
        except Exception as e:
            log(f"[WARN] 30m scoring failed: {e}")
            return pd.Series(dtype=float, index=df_30m.index)

    def score_daily_universe(self, df_1d):
        """Scores the Daily universe dataframe."""
        if not hasattr(self, 'tf_daily_long') or not self.tf_daily_long or df_1d.empty:
            return pd.Series(dtype=float)
        try:
            missing = [c for c in self.tf_daily_features if c not in df_1d.columns]
            if missing:
                missing_df = pd.DataFrame(0.0, index=df_1d.index, columns=missing)
                df_1d = pd.concat([df_1d, missing_df], axis=1)
            
            X_clean = np.nan_to_num(df_1d[self.tf_daily_features].values)
            
            if hasattr(self, 'tf_daily_scaler') and self.tf_daily_scaler is not None:
                try:
                    X_final = self.tf_daily_scaler.transform(X_clean)
                except Exception as e:
                    log(f"[WARN] 1d scaler failed: {e}")
                    X_final = X_clean
            else:
                X_final = X_clean
                
            dmatrix = xgb.DMatrix(X_final, feature_names=self.tf_daily_features)
            l = self.tf_daily_long.predict(dmatrix)
            s = self.tf_daily_short.predict(dmatrix)
            return pd.Series((l - np.mean(l)) - (s - np.mean(s)), index=df_1d.index)
        except Exception as e:
            log(f"[WARN] 1d scoring failed: {e}")
            return pd.Series(dtype=float, index=df_1d.index)
