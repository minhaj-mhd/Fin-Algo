import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_features():
    """
    Returns a dummy 86-feature dictionary simulating the v8_upstox_3y schema.
    Includes normal features, relative/cross-sectional features, and categorical data.
    """
    features = {f"feature_{i}": np.random.randn() for i in range(1, 83)}
    
    # Critical cross-sectional features (Features 83-86)
    features["Market_Mean_Return"] = 0.005
    features["Relative_Return"] = 0.015
    features["Market_Mean_Volatility"] = 0.01
    features["Relative_Volatility"] = 1.5
    
    # Other important meta/raw features not in the 86 strictly but passed around
    features["Close"] = 100.0
    features["HL_Range"] = 0.02 # 2% ATR
    features["Sector"] = "IT"
    features["Volume"] = 150000
    
    return features

@pytest.fixture
def mock_upstox_response():
    """Returns a dummy Upstox API order response."""
    return {
        "status": "success",
        "data": {
            "order_id": "ORD123456789",
            "status": "complete",
            "average_price": 100.50,
            "quantity": 10
        }
    }
