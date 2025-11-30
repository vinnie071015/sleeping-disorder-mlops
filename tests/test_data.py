# tests/test_data.py

import pandas as pd
import sys
import os
import pytest

# 确保能导入 src 目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import clean_data

def test_column_cleaning():
    """测试列名是否被正确标准化（去除空格、转小写）"""
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
    """测试 BMI 类别是否被统一 (Normal Weight -> Normal)"""
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
    """测试清洗函数能正确填充缺失值 (Imputation)"""
    # Arrange: 模拟带有 NaN 值的输入
    mock_data = pd.DataFrame({
        'Person ID': [1, 2],
        'Sleep Duration': [8.0, float('nan')],  # 数值缺失 -> 应填中位数 (8.0)
        'BMI Category': ['Normal Weight', None]  # 分类缺失 -> 应填 'Missing'
    })
    
    # Act
    cleaned_df = clean_data(mock_data)
    
    # Assert
    # 1. 检查列名标准化
    assert 'person_id' in cleaned_df.columns
    
    # 2. 检查数值填充: NaN 应该被填为 8.0 (因为 8.0 的中位数是 8.0)
    # 现在的断言是：它不应该是 NaN
    assert not pd.isna(cleaned_df.loc[1, 'sleep_duration'])
    assert cleaned_df.loc[1, 'sleep_duration'] == 8.0
    
    # 3. 检查分类填充: None 应该被填为 'Missing'
    assert cleaned_df.loc[1, 'bmi_category'] == 'Missing'

def test_handles_empty_input():
    """测试函数收到 None 或空 DataFrame 时的安全退出机制。"""
    # Arrange/Act
    result_none = clean_data(None)
    
    # Arrange/Act
    empty_data = pd.DataFrame()
    result_empty = clean_data(empty_data)
    
    # Assert
    assert result_none is None
    assert result_empty.shape == (0, 0)