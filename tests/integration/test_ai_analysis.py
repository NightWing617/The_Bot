import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from data_ingestion.pdf_parser import parse_racecard
from preprocessing.clean_data import clean_race_data
from preprocessing.feature_engineering import engineer_features
from predictive_modeling.model_train import train_model
from predictive_modeling.predict import predict_outcomes
from betting.kelly_calculator import calculate_kelly_bets
from explainability.shap_explainer import explain_predictions
from explainability.nlp_betting_summary import generate_summary
from tests.utils.test_data_generator import create_test_racecard_pdf, cleanup_test_files

@pytest.fixture
def sample_racecard_pdf():
    """Create and provide a test race card PDF."""
    pdf_path = create_test_racecard_pdf()
    yield pdf_path
    cleanup_test_files(pdf_path)

@pytest.fixture
def sample_historical_data():
    """Create sample historical race data for model training."""
    return pd.DataFrame({
        'horse': ['Horse A', 'Horse B', 'Horse C'] * 10,
        'age': np.random.randint(3, 10, 30),
        'weight': np.random.normal(500, 50, 30),
        'odds': np.random.uniform(2, 10, 30),
        'form': ['1-2-3', '2-1-4', '3-3-1'] * 10,
        'won': np.random.randint(0, 2, 30)
    })

def test_full_ai_analysis_pipeline(sample_racecard_pdf, sample_historical_data):
    """Test the entire AI analysis pipeline from PDF to betting recommendations."""
    try:
        # Step 1: Parse race card PDF
        raw_data = parse_racecard(str(sample_racecard_pdf))
        assert isinstance(raw_data, dict)
        assert 'raw_text' in raw_data
        assert len(raw_data['raw_text']) > 0
        
        # Step 2: Clean and preprocess data
        cleaned_data = clean_race_data(raw_data)
        assert not cleaned_data.empty
        assert 'horse' in cleaned_data.columns
        assert len(cleaned_data) >= 3  # Should have at least 3 horses from our test PDF
        
        features_df = engineer_features(cleaned_data)
        assert 'recent_form' in features_df.columns
        assert 'age_factor' in features_df.columns
        assert 'weight_normalized' in features_df.columns
        
        # Step 3: Train model on historical data
        X_hist = engineer_features(sample_historical_data)
        y_hist = sample_historical_data['won']
        
        feature_cols = ['recent_form', 'age_factor', 'weight_normalized', 'odds_normalized']
        model = train_model(X_hist[feature_cols], y_hist, save_path=None)
        
        # Step 4: Generate predictions
        predictions = predict_outcomes(model, features_df[feature_cols])
        assert len(predictions) == len(features_df)
        assert all('probability' in p for p in predictions)
        assert all('odds' in p for p in predictions)
        
        # Step 5: Calculate optimal bets
        bets = calculate_kelly_bets(predictions, bankroll=1000.0)
        assert len(bets) == len(predictions)
        assert all('stake' in b for b in bets)
        
        # Verify total stakes don't exceed bankroll
        total_stake = sum(b['stake'] for b in bets)
        assert total_stake <= 1000.0
        
        # Step 6: Generate explanations
        explanations = explain_predictions(
            model,
            features_df[feature_cols],
            horse_names=features_df['horse'].tolist()
        )
        
        assert len(explanations) == len(features_df)
        assert all('feature_importance' in e for e in explanations.values())
        
        # Step 7: Generate betting summary
        summary = generate_summary(bets, explanations, bankroll=1000.0)
        assert isinstance(summary, str)
        assert "Race Analysis Summary" in summary
        assert "Individual Horse Analysis" in summary
        
        # Check key insights
        assert any(
            horse in summary 
            for horse in ['Horse A', 'Horse B', 'Horse C']
        )
        assert "Win Probability" in summary
        assert "Recommended Stake" in summary
        
    except Exception as e:
        pytest.fail(f"AI analysis pipeline test failed: {str(e)}")

def test_pipeline_error_handling(sample_racecard_pdf):
    """Test error handling throughout the AI analysis pipeline."""
    # Test with invalid PDF path
    with pytest.raises(Exception):
        parse_racecard("nonexistent.pdf")
    
    # Test with invalid raw data
    with pytest.raises(ValueError):
        clean_race_data({'invalid': 'data'})
    
    # Test with missing features
    invalid_df = pd.DataFrame({'horse': ['A'], 'odds': [2.0]})
    with pytest.raises(ValueError):
        engineer_features(invalid_df)
    
    # Test with invalid probabilities
    invalid_preds = [{'horse': 'A', 'probability': 1.5, 'odds': 2.0}]
    with pytest.raises(ValueError):
        calculate_kelly_bets(invalid_preds)

def test_explanation_consistency(sample_racecard_pdf, sample_historical_data):
    """Test consistency between predictions, explanations, and summaries."""
    # Get real race data
    raw_data = parse_racecard(str(sample_racecard_pdf))
    cleaned_data = clean_race_data(raw_data)
    features_df = engineer_features(cleaned_data)
    
    # Train model
    X_hist = engineer_features(sample_historical_data)
    y_hist = sample_historical_data['won']
    feature_cols = ['recent_form', 'age_factor', 'weight_normalized', 'odds_normalized']
    model = train_model(X_hist[feature_cols], y_hist, save_path=None)
    
    # Generate predictions and explanations
    predictions = predict_outcomes(model, features_df[feature_cols])
    explanations = explain_predictions(
        model,
        features_df[feature_cols],
        horse_names=features_df['horse'].tolist()
    )
    
    # Verify alignment
    for pred in predictions:
        horse = pred['horse']
        assert horse in explanations
        
        # Check that explanation aligns with prediction confidence
        expl = explanations[horse]
        if pred['probability'] > 0.5:
            assert any(
                value > 0 
                for value in expl['feature_importance'].values()
            )
        
    # Verify summary includes key information
    summary = generate_summary(predictions, explanations)
    for pred in predictions[:3]:  # Top 3 horses
        assert pred['horse'] in summary
        assert f"{pred['probability']:.1%}" in summary