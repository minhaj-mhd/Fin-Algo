from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import json

@dataclass(frozen=True)
class DatasetSpec:
    path: str
    label_col: str
    bar_minutes: int
    bar_label_side: str
    label_horizon_bars: int
    label_may_cross_session: bool
    qid_col: str = "Query_ID"
    ticker_col: str = "Ticker"
    datetime_col: str = "DateTime"
    session_close: str = "15:30"
    raw_close_col: Optional[str] = "Close"
    feature_pipeline: Optional[str] = "ranking_v3"
    prefix_invariance_waiver_reason: Optional[str] = None
    raw_source_glob: Optional[str] = None
    unverified_label_waiver_reason: Optional[str] = None

@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter: str
    params: Dict[str, Any]
    features: List[str]
    sides: Tuple[str, ...] = ("long", "short")
    num_boost_round: int = 500
    early_stopping_rounds: int = 50

@dataclass(frozen=True)
class GauntletConfig:
    min_train_months: int = 18
    test_horizon_months: int = 2
    step_months: int = 4
    embargo_bars: Optional[int] = None
    costs_bps: Tuple[float, float] = (6.0, 10.0)
    binding_cost_bps: float = 10.0
    top_k: Tuple[int, ...] = (1, 3)
    primary_k: int = 3
    recent_window_months: int = 12
    trigger_min_net_bps: float = 2.0
    trigger_min_t: float = 2.0
    filter_min_rho_p: float = 0.01
    filter_min_recent_z: float = 2.0
    seed: int = 42
    tod_diagnostic_only: bool = True
