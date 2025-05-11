# feature_engineering.py

import pandas as pd
import numpy as np
from utils.logger import get_logger
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

logger = get_logger(__name__)

def validate_features(df: pd.DataFrame) -> None:
    """Validate required columns and data types."""
    # Check required columns
    required_columns = ['horse', 'odds']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data types
    if 'form' in df.columns and not df['form'].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("Form values must be strings")
    
    if 'age' in df.columns and not df['age'].apply(lambda x: isinstance(x, (int, float))).all():
        raise ValueError("Age values must be numeric")
    
    if 'weight' in df.columns and not df['weight'].apply(lambda x: isinstance(x, (int, float))).all():
        raise ValueError("Weight values must be numeric")
        
    if not df['odds'].apply(lambda x: isinstance(x, (int, float))).all():
        raise ValueError("Odds values must be numeric")

def parse_form(form_string: str) -> List[int]:
    """Parse form string into numeric values."""
    try:
        # Convert form string (e.g. "1-2-3") into list of integers
        results = [int(x) for x in form_string.split('-') if x.isdigit()]
        if not results:
            return [0, 0, 0]  # Default for invalid form
        return results + [0] * (3 - len(results))  # Pad to length 3
    except Exception:
        return [0, 0, 0]  # Default for any parsing error

def calculate_form_rating(form: str) -> float:
    """Calculate rating from recent form string.
    
    Args:
        form: String of form numbers e.g. "1-2-3"
        
    Returns:
        Float rating between 0 and 1
    """
    try:
        if not form or form == '0-0-0':
            return 0.0
            
        # Split form and reverse (most recent first)
        results = form.split('-')[:3]
        results.reverse()
        
        # Calculate weighted average (most recent counts more)
        weights = [0.5, 0.3, 0.2]  # Must sum to 1
        total = 0.0
        
        for i, result in enumerate(results):
            # Convert P (pulled up) to 0
            if result == 'P':
                value = 0
            else:
                try:
                    value = int(result)
                except ValueError:
                    value = 0
                    
            # Higher places get lower scores (1st = 1.0, 2nd = 0.8, etc)
            score = max(0, 1.0 - ((value - 1) * 0.2))
            total += score * weights[i]
            
        return total
        
    except Exception as e:
        logger.error(f"Error calculating form rating: {str(e)}")
        return 0.0

def calculate_age_factor(age: float) -> float:
    """Calculate age-based performance factor.
    
    Args:
        age: Horse's age in years
        
    Returns:
        Float factor between 0 and 1
    """
    try:
        # Peak performance age range is 4-6 years
        if 4 <= age <= 6:
            return 1.0
        elif age < 4:
            return max(0, 0.5 + (age - 2) * 0.25)  # Linear increase
        else:
            return max(0, 1.0 - (age - 6) * 0.1)  # Gradual decline
            
    except Exception as e:
        logger.error(f"Error calculating age factor: {str(e)}")
        return 0.5

def normalize_weight(weight: float, weights: List[float]) -> float:
    """Normalize weight relative to field.
    
    Args:
        weight: Horse's weight
        weights: List of all weights in race
        
    Returns:
        Float between 0 and 1
    """
    try:
        min_weight = min(weights)
        max_weight = max(weights)
        if max_weight == min_weight:
            return 0.5
            
        return (weight - min_weight) / (max_weight - min_weight)
        
    except Exception as e:
        logger.error(f"Error normalizing weight: {str(e)}")
        return 0.5

def normalize_odds(odds: float, all_odds: List[float]) -> float:
    """Convert odds to normalized probability.
    
    Args:
        odds: Horse's decimal odds
        all_odds: List of all odds in race
        
    Returns:
        Float between 0 and 1
    """
    try:
        prob = 1 / odds
        total_prob = sum(1/o for o in all_odds)
        return prob / total_prob if total_prob > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error normalizing odds: {str(e)}")
        return 0.0

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw race data.
    
    Args:
        df: DataFrame with raw race data
        
    Returns:
        DataFrame with engineered features
    """
    try:
        logger.info("Starting feature engineering...")
        validate_features(df)  # Validate input data
        
        result = df.copy()
        
        # Process form data
        logger.info("Processing form data...")
        if 'form' in df.columns:
            result['recent_form'] = df['form'].apply(calculate_form_rating)
        else:
            result['recent_form'] = 0.5  # Default neutral value
            
        # Process age factors
        logger.info("Processing age factors...")
        if 'age' in df.columns:
            result['age_factor'] = df['age'].apply(calculate_age_factor)
        else:
            result['age_factor'] = 0.5
            
        # Process weight data
        logger.info("Processing weight data...")
        if 'weight' in df.columns:
            weights = df['weight'].dropna().tolist()
            if weights:
                result['weight_normalized'] = df['weight'].apply(
                    lambda w: normalize_weight(w, weights)
                )
            else:
                result['weight_normalized'] = 0.5
        else:
            result['weight_normalized'] = 0.5
            
        # Process odds data
        if 'odds' in df.columns:
            odds = df['odds'].dropna().tolist()
            if odds:
                result['odds_normalized'] = df['odds'].apply(
                    lambda o: normalize_odds(o, odds)
                )
            else:
                result['odds_normalized'] = 0.5
                
        logger.info("Feature engineering completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def select_features(df, target_cols=None):
    """
    Select relevant features for model training.
    
    Args:
        df: DataFrame with engineered features
        target_cols: List of specific columns to select
    
    Returns:
        DataFrame with selected features
    """
    default_features = [
        'recent_form',
        'age_factor',
        'weight_normalized',
        'odds_normalized'
    ]
    
    try:
        features = target_cols if target_cols else default_features
        missing_cols = [col for col in features if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        return df[features]
    
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise