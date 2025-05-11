# predict.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_input_features(features_df, model):
    """Validate that input features match model requirements."""
    if not hasattr(model, 'feature_names_in_'):
        logger.warning("Model doesn't have feature_names_in_ attribute. Skipping feature validation.")
        return True
        
    missing_features = set(model.feature_names_in_) - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    return True

def get_prediction_uncertainty(probabilities):
    """Calculate prediction uncertainty using entropy."""
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
    max_entropy = -np.log2(1/len(probabilities[0]))  # Maximum possible entropy
    uncertainty = entropy / max_entropy  # Normalize to [0,1]
    return uncertainty

def predict_outcomes(model, features_df, cv_folds=5):
    """
    Generate predictions with uncertainty estimates and cross-validation.
    
    Args:
        model: Trained model instance
        features_df: DataFrame of features
        cv_folds: Number of cross-validation folds
    
    Returns:
        List of dictionaries containing predictions and metadata
    """
    try:
        logger.info("Starting prediction pipeline...")
        
        # Validate input features
        validate_input_features(features_df, model)
        
        # Standardize features if model expects it
        if hasattr(model, 'preprocessing') and model.preprocessing.get('standardize', False):
            scaler = StandardScaler()
            features = scaler.fit_transform(features_df)
            features = pd.DataFrame(features, columns=features_df.columns)
        else:
            features = features_df.copy()
        
        # Get cross-validated predictions
        cv_probas = cross_val_predict(
            model, 
            features, 
            cv=cv_folds,
            method='predict_proba'
        )
        
        # Get final model predictions
        final_probas = model.predict_proba(features)
        
        # Calculate uncertainties
        uncertainties = get_prediction_uncertainty(final_probas)
        
        # Calculate prediction variance across CV folds
        cv_variance = np.std(cv_probas, axis=0)[:,1]  # Variance of positive class prob
        
        outcomes = []
        for i, row in features_df.iterrows():
            win_prob = final_probas[i][1]  # Probability of positive class
            cv_var = cv_variance[i]
            uncertainty = uncertainties[i]
            
            prediction = {
                'horse': row['horse'] if 'horse' in row else f"Horse_{i}",
                'probability': round(float(win_prob), 4),
                'cv_variance': round(float(cv_var), 4),
                'uncertainty': round(float(uncertainty), 4),
                'odds': round(float(row['odds']) if 'odds' in row else 1/win_prob, 2)
            }
            
            # Add confidence level based on uncertainty
            if uncertainty < 0.2:
                prediction['confidence'] = 'very_high'
            elif uncertainty < 0.4:
                prediction['confidence'] = 'high'
            elif uncertainty < 0.6:
                prediction['confidence'] = 'moderate'
            else:
                prediction['confidence'] = 'low'
            
            outcomes.append(prediction)
        
        # Sort by probability
        outcomes.sort(key=lambda x: x['probability'], reverse=True)
        
        # Log prediction summary
        logger.info(f"Generated predictions for {len(outcomes)} horses")
        logger.info(f"Top prediction: {outcomes[0]['horse']} with {outcomes[0]['probability']:.2%} probability")
        
        return outcomes
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def calibrate_probabilities(predictions, historical_data=None):
    """
    Calibrate predicted probabilities using historical performance.
    
    Args:
        predictions: List of prediction dictionaries
        historical_data: Optional DataFrame of historical predictions and outcomes
    
    Returns:
        List of predictions with calibrated probabilities
    """
    try:
        if historical_data is None:
            logger.warning("No historical data provided for calibration")
            return predictions
            
        # TODO: Implement probability calibration using historical data
        # This could use isotonic regression or Platt scaling
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error during probability calibration: {str(e)}")
        return predictions  # Return uncalibrated predictions on error
