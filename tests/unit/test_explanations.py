import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainability.shap_explainer import ModelExplainer, explain_predictions
from explainability.nlp_betting_summary import generate_summary, BettingSummaryGenerator

@pytest.fixture
def sample_model():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def sample_features():
    return pd.DataFrame({
        'recent_form': np.random.rand(5),
        'age_factor': np.random.rand(5),
        'weight_normalized': np.random.rand(5),
        'odds_normalized': np.random.rand(5)
    })

@pytest.fixture
def sample_predictions():
    return [
        {
            'horse': 'Horse A',
            'probability': 0.85,
            'odds': 2.5,
            'stake': 100.0,
            'kelly_value': 0.25
        },
        {
            'horse': 'Horse B',
            'probability': 0.35,
            'odds': 4.0,
            'stake': 50.0,
            'kelly_value': 0.15
        },
        {
            'horse': 'Horse C',
            'probability': 0.15,
            'odds': 8.0,
            'stake': 20.0,
            'kelly_value': 0.05
        }
    ]

def test_model_explainer_initialization(sample_model):
    explainer = ModelExplainer(sample_model)
    assert explainer.model == sample_model
    assert explainer.explainer is None

def test_explanation_generation(sample_model, sample_features):
    explainer = ModelExplainer(sample_model)
    explanations = explainer.generate_explanations(
        sample_features,
        horse_names=['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E']
    )
    
    assert isinstance(explanations, dict)
    assert len(explanations) == len(sample_features)
    
    for horse, expl in explanations.items():
        assert 'feature_importance' in expl
        assert 'base_value' in expl
        assert isinstance(expl['feature_importance'], dict)
        assert len(expl['feature_importance']) == len(sample_features.columns)

def test_natural_language_explanation(sample_model, sample_features):
    explainer = ModelExplainer(sample_model)
    explanations = explainer.generate_explanations(sample_features)
    
    for horse, expl in explanations.items():
        nl_explanation = explainer.generate_natural_language_explanation(
            horse, expl
        )
        assert isinstance(nl_explanation, str)
        assert len(nl_explanation) > 0

def test_betting_summary_generation(sample_predictions):
    generator = BettingSummaryGenerator()
    summary = generator.generate_summary(
        sample_predictions,
        bankroll=1000.0
    )
    
    assert isinstance(summary, str)
    assert "Race Analysis Summary" in summary
    assert "Individual Horse Analysis" in summary
    assert "Betting Strategy" in summary
    
    # Check key components
    assert "Horse A" in summary
    assert "Win Probability" in summary
    assert "Recommended Stake" in summary
    assert "Total Recommended Stake" in summary

def test_betting_summary_insights(sample_predictions):
    generator = BettingSummaryGenerator()
    insights = generator._generate_key_insights(sample_predictions)
    
    assert isinstance(insights, list)
    assert len(insights) > 0
    assert any("strong favorites" in insight.lower() for insight in insights)
    assert any("kelly" in insight.lower() for insight in insights)

def test_betting_summary_edge_cases():
    generator = BettingSummaryGenerator()
    
    # Test empty predictions
    summary = generator.generate_summary([])
    assert isinstance(summary, str)
    assert "Error" not in summary
    
    # Test missing probabilities
    bad_predictions = [{'horse': 'Horse X'}]
    summary = generator.generate_summary(bad_predictions)
    assert isinstance(summary, str)
    assert "Error" not in summary

def test_end_to_end_explanation(sample_model, sample_features):
    horse_names = ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E']
    explanations = explain_predictions(sample_model, sample_features, horse_names)
    
    assert isinstance(explanations, dict)
    assert len(explanations) == len(horse_names)
    
    for horse in horse_names:
        assert horse in explanations
        assert 'natural_language' in explanations[horse]
        assert 'feature_importance' in explanations[horse]