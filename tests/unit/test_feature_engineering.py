import pytest
import pandas as pd
from preprocessing.feature_engineering import engineer_features

def test_feature_engineering_output(sample_race_data):
    # Test that features are added correctly
    result = engineer_features(sample_race_data.copy())
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert 'recent_form' in result.columns
    
    # Check data validity
    assert result['recent_form'].between(0, 1).all()
    
def test_feature_engineering_form_calculation(sample_race_data):
    # Test form calculation logic
    df = sample_race_data.copy()
    result = engineer_features(df)
    
    # Verify form calculation (assuming higher is better)
    horse_c_index = df[df['horse'] == 'Horse C'].index[0]
    horse_b_index = df[df['horse'] == 'Horse B'].index[0]
    
    # Horse C (3-3-1) should have better recent form than Horse B (2-1-4)
    assert result.loc[horse_c_index, 'recent_form'] > result.loc[horse_b_index, 'recent_form']

def test_engineer_features():
    # Test with valid input
    input_df = pd.DataFrame({
        'horse': ['Horse A', 'Horse B'],
        'age': [5, 6],
        'weight': [500, 520],
        'odds': [4.5, 6.0],
        'form': ['1-2-3', '2-1-4']
    })
    
    result = engineer_features(input_df)
    assert 'recent_form' in result.columns
    assert 'age_factor' in result.columns
    assert 'weight_normalized' in result.columns
    assert len(result) == len(input_df)

def test_missing_required_columns():
    # Test with missing required columns
    invalid_df = pd.DataFrame({
        'horse': ['Horse A', 'Horse B']  # Missing required columns
    })
    
    with pytest.raises(ValueError):
        engineer_features(invalid_df)

def test_invalid_data_types():
    # Test with invalid data types
    invalid_df = pd.DataFrame({
        'horse': ['Horse A', 'Horse B'],
        'age': ['five', 'six'],  # Invalid age values
        'weight': [500, 520],
        'odds': [4.5, 6.0],
        'form': ['1-2-3', '2-1-4']
    })
    
    with pytest.raises(ValueError):
        engineer_features(invalid_df)

def test_feature_ranges():
    # Test that normalized features are in valid ranges
    input_df = pd.DataFrame({
        'horse': ['Horse A', 'Horse B'],
        'age': [5, 6],
        'weight': [500, 520],
        'odds': [4.5, 6.0],
        'form': ['1-2-3', '2-1-4']
    })
    
    result = engineer_features(input_df)
    
    # Check ranges
    assert all(0 <= x <= 1 for x in result['recent_form'])
    assert all(0 <= x <= 1 for x in result['age_factor'])
    assert all(0 <= x <= 1 for x in result['weight_normalized'])
    assert all(0 <= x <= 1 for x in result['odds_normalized'])