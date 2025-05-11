import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from preprocessing.clean_data import clean_race_data
from preprocessing.feature_engineering import engineer_features
from predictive_modeling.model_train import train_model
from predictive_modeling.predict import predict_outcomes
from betting.kelly_calculator import calculate_kelly_bets
from explainability.shap_explainer import explain_predictions
from interface.app import present_results
from realtime_adapter.live_odds_monitor import update_with_live_data

@pytest.fixture
def sample_config():
    return {
        'bankroll': 1000.0,
        'kelly_fraction': 0.5,
        'model_path': 'test_model.pkl'
    }

@pytest.fixture
def sample_raw_data():
    return {
        'raw_text': """
        Horse A
        Age: 5
        Weight: 500
        Odds: 4.5
        Form: 1-2-3
        
        Horse B
        Age: 6
        Weight: 520
        Odds: 6.0
        Form: 2-1-4
        """
    }

def test_full_pipeline_integration(sample_raw_data, sample_config):
    """Test the entire pipeline from raw data to betting recommendations."""
    try:
        # Step 1: Clean data
        cleaned_data = clean_race_data(sample_raw_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.empty
        assert 'horse' in cleaned_data.columns
        assert 'odds' in cleaned_data.columns
        
        # Step 2: Engineer features
        features_df = engineer_features(cleaned_data)
        assert isinstance(features_df, pd.DataFrame)
        assert 'recent_form' in features_df.columns
        
        # Step 3: Train model with dummy target
        X = features_df[['recent_form', 'age_factor', 'weight_normalized']]
        y = np.random.randint(0, 2, size=len(X))  # Dummy binary outcome
        model = train_model(X, y, save_path=None)
        
        # Step 4: Generate predictions
        predictions = predict_outcomes(model, X)
        assert isinstance(predictions, list)
        assert len(predictions) == len(X)
        assert all('probability' in p for p in predictions)
        assert all('odds' in p for p in predictions)
        
        # Step 5: Calculate bets
        bets = calculate_kelly_bets(
            predictions,
            bankroll=sample_config['bankroll'],
            kelly_fraction=sample_config['kelly_fraction']
        )
        assert isinstance(bets, list)
        assert len(bets) == len(predictions)
        assert all('stake' in b for b in bets)
        
        # Verify total stakes don't exceed bankroll
        total_stake = sum(b['stake'] for b in bets)
        assert total_stake <= sample_config['bankroll']
        
    except Exception as e:
        pytest.fail(f"Pipeline integration test failed: {str(e)}")

def test_pipeline_error_handling(sample_raw_data):
    """Test error handling throughout the pipeline."""
    # Test with invalid raw data
    with pytest.raises(TypeError):
        clean_race_data("invalid input")
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'horse': ['A', 'B']})  # Missing odds
    with pytest.raises(ValueError):
        engineer_features(invalid_df)
    
    # Test with invalid odds values
    invalid_odds_df = pd.DataFrame({
        'horse': ['A', 'B'],
        'odds': [0.5, -1]  # Invalid odds values
    })
    with pytest.raises(ValueError):
        clean_race_data({'raw_text': str(invalid_odds_df)})

def test_model_persistence(sample_raw_data, sample_config):
    """Test model saving and loading functionality."""
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp_model:
        # Create and save model
        cleaned_data = clean_race_data(sample_raw_data)
        features_df = engineer_features(cleaned_data)
        X = features_df[['recent_form', 'age_factor']]
        y = np.random.randint(0, 2, size=len(X))
        
        model = train_model(X, y, save_path=tmp_model.name)
        
        # Verify model was saved
        assert Path(tmp_model.name).exists()
        
        # Test predictions with loaded model
        predictions = predict_outcomes(model, X)
        assert isinstance(predictions, list)
        assert len(predictions) > 0