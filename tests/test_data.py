# tests/test_data.py

import pandas as pd
import sys
import os
import pytest

# Ensure modules in the src directory can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import clean_data

def test_column_cleaning():
    """Test if column names are correctly standardized (removing spaces, converting to lowercase)"""
    # Arrange
    mock_data = pd.DataFrame({
        'Person ID': [1],
        'Sleep Duration': [8.5],
        'BMI Category': ['Normal Weight']
    })
    
    # Act
    cleaned_df = clean_data(mock_data)
    
    # Assert
    assert 'person_id' in cleaned_df.columns
    assert 'sleep_duration' in cleaned_df.columns
    assert 'BMI Category' not in cleaned_df.columns

def test_bmi_category_normalization():
    """Test if BMI categories are unified (Normal Weight -> Normal)"""
    # Arrange
    mock_data = pd.DataFrame({
        'BMI Category': ['Normal', 'Normal Weight', 'Obese']
    })
    
    # Act
    cleaned_df = clean_data(mock_data)
    
    # Assert
    unique_values = cleaned_df['bmi_category'].unique()
    assert 'Normal Weight' not in unique_values
    assert 'Normal' in unique_values
    assert 'Obese' in unique_values

def test_handles_missing_values():
    """Test if the cleaning function correctly imputes missing values (Imputation)"""
    # Arrange: Simulate input with NaN values
    mock_data = pd.DataFrame({
        'Person ID': [1, 2],
        'Sleep Duration': [8.0, float('nan')],  # Numerical missing -> should be imputed with median (8.0)
        'BMI Category': ['Normal Weight', None]  # Categorical missing -> should be imputed with 'Missing'
    })
    
    # Act
    cleaned_df = clean_data(mock_data)
    
    # Assert
    # 1. Check column name standardization
    assert 'person_id' in cleaned_df.columns
    
    # 2. Check numerical imputation: NaN should be filled with 8.0 (since the median of 8.0 is 8.0)
    # The current assertion is: it should not be NaN
    assert not pd.isna(cleaned_df.loc[1, 'sleep_duration'])
    assert cleaned_df.loc[1, 'sleep_duration'] == 8.0
    
    # 3. Check categorical imputation: None should be filled with 'Missing'
    assert cleaned_df.loc[1, 'bmi_category'] == 'Missing'

def test_handles_empty_input():
    """Test the function's safe exit mechanism when receiving None or an empty DataFrame."""
    # Arrange/Act
    result_none = clean_data(None)
    
    # Arrange/Act
    empty_data = pd.DataFrame()
    result_empty = clean_data(empty_data)
    
    # Assert
    assert result_none is None
    assert result_empty.shape == (0, 0)