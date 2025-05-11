import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_race_data():
    return pd.DataFrame({
        'horse': ['Horse A', 'Horse B', 'Horse C'],
        'age': [5, 6, 4],
        'weight': [500, 520, 490],
        'jockey': ['Smith', 'Jones', 'Brown'],
        'odds': [4.5, 6.0, 3.2],
        'form': ['1-2-3', '2-1-4', '3-3-1']
    })

@pytest.fixture
def sample_predictions():
    return [
        {'horse': 'Horse A', 'probability': 0.35, 'odds': 4.5},
        {'horse': 'Horse B', 'probability': 0.25, 'odds': 6.0},
        {'horse': 'Horse C', 'probability': 0.40, 'odds': 3.2}
    ]