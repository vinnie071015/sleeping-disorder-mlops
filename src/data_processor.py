"""
Data processing module for sleep disorder prediction.
Handles data loading and cleaning/preprocessing.
"""
import os
import pandas as pd

def load_data(file_path):
    """
    读取 CSV 数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 找不到文件 {file_path}")

    try:
        print(f"--- INFO: Loading data from {file_path} ---")
        df = pd.read_csv(file_path)
        print(f"--- INFO: Data loaded. Shape: {df.shape} ---")
        return df
    except Exception as e:
        # 使用 'from e' 保留原始错误堆栈，使用 RuntimeError 代替通用 Exception
        raise RuntimeError(f"读取文件时发生错误: {e}") from e

def clean_data(df):
    """
    数据清洗逻辑：
    1. 标准化列名：转小写，空格替换为下划线。
    2. 缺失值填充。
    3. 处理 'BMI Category' 中的不一致。
    """
    if df is None:
        return None

    df_clean = df.copy()

    # 1. 列名标准化
    df_clean.columns = [
        c.lower().strip().replace(' ', '_').replace('/', '_')
        for c in df_clean.columns
    ]
    print("--- DEBUG: Columns standardized. ---")

    # --- 2. 缺失值填充 (Imputation) ---
    numerical_cols = df_clean.select_dtypes(include=['number']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns

    print(f"--- DEBUG: Imputing {len(numerical_cols)} numerical features (Median) ---")
    df_clean[numerical_cols] = df_clean[numerical_cols].fillna(
        df_clean[numerical_cols].median()
    )

    print(f"--- DEBUG: Imputing {len(categorical_cols)} categorical features ('Missing') ---")
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Missing')

    # --- 3. 特定清洗 ---
    if 'bmi_category' in df_clean.columns:
        df_clean['bmi_category'] = df_clean['bmi_category'].replace('Normal Weight', 'Normal')
        print("--- DEBUG: BMI categories normalized. ---")

    print(f"--- INFO: Data cleaning completed. Final shape: {df_clean.shape} ---")
    return df_clean