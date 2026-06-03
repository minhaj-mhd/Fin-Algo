import pytest
import pandas as pd
import numpy as np
from scripts.vanguard.model_inference import ModelManager

def test_score_universe_zscore_exclusion(mocker):
    # Mock the XGBoost boosters to bypass model loading and predicting
    mock_booster = mocker.patch("xgboost.Booster")
    mock_predict = mock_booster.return_value.predict
    # Mock to return a dummy score array of length 2
    mock_predict.return_value = np.array([0.5, 0.2])
    
    manager = ModelManager()
    manager.bst_long = mock_booster()
    manager.bst_short = mock_booster()
    
    # Define our feature columns
    manager.feature_cols = ["Standard_Feature", "Market_Mean_Return", "Relative_Return"]
    
    # Create a dummy dataframe with 2 rows to allow std dev calculation
    df = pd.DataFrame({
        "ticker": ["RELIANCE", "TCS"],
        "Standard_Feature": [10.0, 20.0],
        "Market_Mean_Return": [0.005, 0.005], # This is an excluded feature
        "Relative_Return": [0.02, -0.01]      # This is also an excluded feature
    })
    
    # Keep original values to check exclusions
    orig_market_mean = df["Market_Mean_Return"].copy()
    orig_relative = df["Relative_Return"].copy()
    
    # Run the scoring mechanism
    result_df = manager.score_universe(df, ticker_metadata={})
    
    assert not result_df.empty
    
    # Verify the non-excluded feature WAS z-scored.
    # Mean of [10, 20] is 15. Std is 7.071. Z-scores are -0.707 and 0.707.
    assert result_df["Standard_Feature"].iloc[0] != 10.0
    
    # Verify the EXCLUDED features were untouched
    assert (result_df["Market_Mean_Return"] == orig_market_mean).all()
    assert (result_df["Relative_Return"] == orig_relative).all()
    
    # Verify the ranking columns were added
    assert "Long_Conviction" in result_df.columns
    assert "Long_Rank" in result_df.columns
