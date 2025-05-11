# model_train.py

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from utils.logger import get_logger

logger = get_logger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series, save_path: str = None) -> RandomForestClassifier:
    """Train a model on the given data.
    
    Args:
        X: Feature matrix
        y: Target vector
        save_path: Optional path to save the model
        
    Returns:
        Trained RandomForestClassifier model
    """
    try:
        logger.info("Training model...")
        
        # Initialize model with optimized hyperparameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Train model on full dataset
        model.fit(X, y)
        
        # Calculate appropriate number of CV folds based on sample size
        n_samples = len(X)
        if n_samples < 5:
            logger.warning("Dataset too small for cross-validation, using train-test split")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            logger.info(f"Test set accuracy: {score:.3f}")
        else:
            n_splits = min(5, n_samples // 2)  # Ensure at least 2 samples per fold
            cv_scores = cross_val_score(model, X, y, cv=n_splits)
            avg_cv_score = np.mean(cv_scores)
            logger.info(f"Cross-validation accuracy ({n_splits}-fold): {avg_cv_score:.3f}")
        
        # Save model if path provided
        if save_path:
            joblib.dump(model, save_path)
            logger.info(f"Model saved to {save_path}")
            
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def load_model(model_path: str) -> RandomForestClassifier:
    """Load a saved model.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded model
    """
    try:
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
