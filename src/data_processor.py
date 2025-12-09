"""
Data processing module for sleep disorder prediction.
Handles data loading and cleaning/preprocessing.
"""
import os
import pandas as pd

def load_data(file_path):
    """
    Reads CSV data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found {file_path}")

    try:
        print(f"--- INFO: Loading data from {file_path} ---")
        df = pd.read_csv(file_path)
        print(f"--- INFO: Data loaded. Shape: {df.shape} ---")
        return df
    except Exception as e:
        # Use 'from e' to preserve the original traceback, using RuntimeError instead of generic Exception
        raise RuntimeError(f"An error occurred while reading the file: {e}") from e

def clean_data(df):
    """
    Data cleaning logic:
    1. Standardize column names: convert to lowercase, replace spaces with underscores.
    2. Missing value imputation.
    3. Handle inconsistencies in 'BMI Category'.
    """
    if df is None:
        return None

    df_clean = df.copy()

    # 1. Column name standardization
    df_clean.columns = [
        c.lower().strip().replace(' ', '_').replace('/', '_')
        for c in df_clean.columns
    ]
    print("--- DEBUG: Columns standardized. ---")

    # --- 2. Missing Value Imputation ---
    numerical_cols = df_clean.select_dtypes(include=['number']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns

    print(f"--- DEBUG: Imputing {len(numerical_cols)} numerical features (Median) ---")
    df_clean[numerical_cols] = df_clean[numerical_cols].fillna(
        df_clean[numerical_cols].median()
    )

    print(f"--- DEBUG: Imputing {len(categorical_cols)} categorical features ('Missing') ---")
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Missing')

    # --- 3. Specific cleaning ---
    if 'bmi_category' in df_clean.columns:
        df_clean['bmi_category'] = df_clean['bmi_category'].replace('Normal Weight', 'Normal')
        print("--- DEBUG: BMI categories normalized. ---")

    print(f"--- INFO: Data cleaning completed. Final shape: {df_clean.shape} ---")
    return df_clean