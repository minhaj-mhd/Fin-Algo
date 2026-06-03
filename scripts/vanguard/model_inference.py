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
            self.tf_15m_long.load_model("models/v1_15min/xgb_long_model.json")
            self.tf_15m_long.set_param({'device': 'cuda'})
            self.tf_15m_short = xgb.Booster()
            self.tf_15m_short.load_model("models/v1_15min/xgb_short_model.json")
            self.tf_15m_short.set_param({'device': 'cuda'})
            with open("models/v1_15min/metadata.json", "r") as f:
                self.tf_15m_features = json.load(f)["features"]
            if os.path.exists("models/v1_15min/scaler.pkl"):
                with open("models/v1_15min/scaler.pkl", "rb") as f:
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

        # Z-Scoring logic
        exclude_from_z = [
            "ticker", "DateTime", "Close", "Open", "High", "Low", "Volume",
            "Market_Mean_Return", "Relative_Return", "Market_Mean_Volatility", "Relative_Volatility",
            "Hour", "DayOfWeek", "Sector",
            "Nifty_1H_Return", "Nifty_3H_Return", "Nifty_5H_Return", "Nifty_RSI", "Nifty_HL_Range", "Nifty_20H_Std",
            "Sector_Mean_Return", "Stock_vs_Sector", "Sector_Breadth", "Sector_Count", "Sector_Volatility", "Sector_Rank",
            "VIX_Level", "VIX_Change", "VIX_5D_MA", "VIX_High", "VIX_Extreme", "Market_Regime",
            "Is_Open_Hour", "Is_Close_Hour", "Time_To_Close", "Up_Streak", "Down_Streak",
            "RSI_14_Raw", "Stoch_K_Raw", "PercentB_Raw", "Daily_RSI", "Daily_SMA20_Dist", "Daily_Trend", "Daily_ATR_Pct",
            "Bar_Position", "Green_Bar_Ratio_5", "Accumulation_5"
        ]

        features_to_zscore = [c for c in self.feature_cols if c not in exclude_from_z]
        for col in features_to_zscore:
            if col in scores_df.columns:
                mean = scores_df[col].mean()
                std = scores_df[col].std()
                if pd.isna(std) or std == 0:
                    scores_df[col] = 0.0
                else:
                    scores_df[col] = (scores_df[col] - mean) / std

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
            long_conviction = long_score - short_score
            short_conviction = short_score - long_score
            
            scores_df = scores_df.assign(
                long_score=long_score,
                short_score=short_score,
                Long_Conviction=long_conviction,
                Short_Conviction=short_conviction,
                Long_Rank=pd.Series(long_conviction).rank(ascending=False).values,
                Short_Rank=pd.Series(short_conviction).rank(ascending=False).values
            )

            # Score 15-Min model
            if hasattr(self, 'tf_15m_long') and self.tf_15m_long:
                try:
                    missing_15 = [c for c in self.tf_15m_features if c not in scores_df.columns]
                    if missing_15:
                        scores_df = scores_df.assign(**{col: 0.0 for col in missing_15})
                    X_15 = scores_df[self.tf_15m_features].values
                    X_15_clean = np.nan_to_num(X_15)
                    
                    scaler_15m_fitted = (
                        self.tf_15m_scaler is not None
                        and hasattr(self.tf_15m_scaler, 'scale_')
                        and self.tf_15m_scaler.scale_ is not None
                    )
                    if scaler_15m_fitted:
                        X_15_final = self.tf_15m_scaler.transform(X_15_clean)
                    else:
                        X_15_final = X_15_clean
                        
                    d_15 = xgb.DMatrix(X_15_final, feature_names=self.tf_15m_features)
                    l_15 = self.tf_15m_long.predict(d_15)
                    s_15 = self.tf_15m_short.predict(d_15)
                    scores_df["score_15m"] = l_15 - s_15
                except Exception as e15:
                    log(f"[WARN] 15m scoring failed: {e15}")

            # Score 30-Min model
            if hasattr(self, 'tf_30m_long') and self.tf_30m_long:
                try:
                    missing_30 = [c for c in self.tf_30m_features if c not in scores_df.columns]
                    if missing_30:
                        scores_df = scores_df.assign(**{col: 0.0 for col in missing_30})
                    X_30 = scores_df[self.tf_30m_features].values
                    X_30_clean = np.nan_to_num(X_30)
                    
                    scaler_30m_fitted = (
                        self.tf_30m_scaler is not None
                        and hasattr(self.tf_30m_scaler, 'scale_')
                        and self.tf_30m_scaler.scale_ is not None
                    )
                    if scaler_30m_fitted:
                        X_30_final = self.tf_30m_scaler.transform(X_30_clean)
                    else:
                        X_30_final = X_30_clean
                        
                    d_30 = xgb.DMatrix(X_30_final, feature_names=self.tf_30m_features)
                    l_30 = self.tf_30m_long.predict(d_30)
                    s_30 = self.tf_30m_short.predict(d_30)
                    scores_df["score_30m"] = l_30 - s_30
                except Exception as e30:
                    log(f"[WARN] 30m scoring failed: {e30}")

            # Score Daily model
            if hasattr(self, 'tf_daily_long') and self.tf_daily_long:
                try:
                    missing_d = [c for c in self.tf_daily_features if c not in scores_df.columns]
                    if missing_d:
                        scores_df = scores_df.assign(**{col: 0.0 for col in missing_d})
                    X_d = scores_df[self.tf_daily_features].values
                    X_d_clean = np.nan_to_num(X_d)
                    
                    scaler_daily_fitted = (
                        self.tf_daily_scaler is not None
                        and hasattr(self.tf_daily_scaler, 'scale_')
                        and self.tf_daily_scaler.scale_ is not None
                    )
                    if scaler_daily_fitted:
                        X_d_final = self.tf_daily_scaler.transform(X_d_clean)
                    else:
                        X_d_final = X_d_clean
                        
                    d_d = xgb.DMatrix(X_d_final, feature_names=self.tf_daily_features)
                    l_d = self.tf_daily_long.predict(d_d)
                    s_d = self.tf_daily_short.predict(d_d)
                    scores_df["score_1d"] = l_d - s_d
                except Exception as ed:
                    log(f"[WARN] 1d scoring failed: {ed}")

            return scores_df

        except Exception as e:
            log(f"[ERROR] Prediction Error: {e}")
            return pd.DataFrame()
