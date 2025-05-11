# clean_data.py

import pandas as pd
import logging
from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_raw_data(raw_data: Dict[str, Any]) -> None:
    """Validate the structure of incoming raw data."""
    if not isinstance(raw_data, dict):
        raise TypeError("Raw data must be a dictionary")
    if 'horses' not in raw_data and 'raw_text' not in raw_data:
        raise KeyError("Raw data missing both 'horses' and 'raw_text' keys")

def extract_horses_from_raw_text(raw_text: str) -> List[Dict[str, Any]]:
    """Extract horse data from raw text format."""
    horses = []
    current_horse = {}
    
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            if current_horse:
                horses.append(current_horse)
                current_horse = {}
            continue
            
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'horse':
                if current_horse:
                    horses.append(current_horse)
                current_horse = {'horse': value}
            else:
                if not current_horse:
                    current_horse = {}
                current_horse[key] = value
                
    if current_horse:
        horses.append(current_horse)
        
    return horses

def clean_horse_data(horse_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean individual horse data entries."""
    cleaned = {}
    
    # Clean and validate horse name (must be present)
    if 'horse' not in horse_data or not horse_data['horse']:
        raise ValueError("Missing horse name")
    cleaned['horse'] = str(horse_data['horse']).strip()
    if not cleaned['horse']:
        raise ValueError("Empty horse name")
    
    # Clean odds (required field)
    if 'odds' not in horse_data:
        raise ValueError(f"Missing odds for horse {cleaned['horse']}")
    try:
        cleaned['odds'] = float(horse_data['odds'])
        if cleaned['odds'] <= 1:
            raise ValueError(f"Invalid odds value: {cleaned['odds']}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid odds for horse {cleaned['horse']}: {str(e)}")
    
    # Clean age if present
    if 'age' in horse_data and horse_data['age']:
        try:
            cleaned['age'] = int(horse_data['age'])
            if not 2 <= cleaned['age'] <= 20:
                logger.warning(f"Unusual age value for {cleaned['horse']}: {cleaned['age']}")
        except (TypeError, ValueError):
            logger.warning(f"Invalid age value for {cleaned['horse']}, setting to None")
            cleaned['age'] = None
    
    # Clean weight if present
    if 'weight' in horse_data and horse_data['weight']:
        try:
            cleaned['weight'] = float(horse_data['weight'])
            if not 300 <= cleaned['weight'] <= 700:
                logger.warning(f"Unusual weight value for {cleaned['horse']}: {cleaned['weight']}")
        except (TypeError, ValueError):
            logger.warning(f"Invalid weight value for {cleaned['horse']}, setting to None")
            cleaned['weight'] = None
    
    # Clean form if present
    if 'form' in horse_data and horse_data['form']:
        form = str(horse_data['form']).strip()
        if form and all(c in '0123456789-P' for c in form):
            cleaned['form'] = form
        else:
            cleaned['form'] = '0-0-0'  # Default form for invalid/missing data
    else:
        cleaned['form'] = '0-0-0'
    
    # Clean jockey if present
    if 'jockey' in horse_data and horse_data['jockey']:
        cleaned['jockey'] = str(horse_data['jockey']).strip()
    
    return cleaned

def clean_race_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Clean and structure race data.
    
    Args:
        raw_data: Raw data from PDF parser
        
    Returns:
        DataFrame with cleaned race data
    """
    try:
        # Validate input
        validate_raw_data(raw_data)
        
        # Get horses data
        horses = []
        if 'horses' in raw_data and raw_data['horses']:
            horses.extend(raw_data['horses'])
        
        # Extract additional horses from raw text if available
        if 'raw_text' in raw_data and raw_data['raw_text']:
            text_horses = extract_horses_from_raw_text(raw_data['raw_text'])
            
            # Merge horses based on names to avoid duplicates
            existing_names = {h['horse'] for h in horses}
            for horse in text_horses:
                if horse['horse'] not in existing_names:
                    horses.append(horse)
                    existing_names.add(horse['horse'])
            
        if not horses:
            raise ValueError("No valid horse data found")
            
        # Convert to DataFrame
        df = pd.DataFrame(horses)
        
        # Ensure required columns exist
        required_cols = ['horse', 'odds']  # Minimum required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean numeric columns
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
        
        # Clean form data
        df['form'] = df['form'].fillna('0-0-0')
        
        # Drop rows with missing critical data
        df = df.dropna(subset=['horse', 'odds'])
        
        if len(df) == 0:
            raise ValueError("No valid horse data after cleaning")
            
        return df
        
    except Exception as e:
        logger.error(f"Error in clean_race_data: {str(e)}")
        raise
