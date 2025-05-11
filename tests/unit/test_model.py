import pytest
import numpy as np
from predictive_modeling.model_train import train_model
from predictive_modeling.predict import predict_outcomes
from predictive_modeling.evaluate import evaluate_model

def test_model_training(sample_race_data):
    # Prepare sample data
    X = sample_race_data[['age', 'weight']]
    y = np.random.randint(0, 2, size=len(X))  # Binary outcome
    
    # Test model training
    model = train_model(X, y, save_path=None)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
def test_prediction_format(sample_race_data):
    # Prepare data
    X = sample_race_data[['age', 'weight']]
    y = np.random.randint(0, 2, size=len(X))
    model = train_model(X, y, save_path=None)
    
    # Test predictions
    predictions = predict_outcomes(model, X)
    assert isinstance(predictions, list)
    assert all(isinstance(p, dict) for p in predictions)
    assert all('probability' in p for p in predictions)
    assert all(0 <= p['probability'] <= 1 for p in predictions)

def test_model_evaluation(sample_race_data):
    # Prepare data
    X = sample_race_data[['age', 'weight']]
    y = np.random.randint(0, 2, size=len(X))
    model = train_model(X, y, save_path=None)
    
    # Test evaluation metrics
    metrics = evaluate_model(model, X, y)
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1