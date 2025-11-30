"""
Smoke tests for all 3 models (LR, SVM, RF).
"""
import sys
import os
import pytest
from unittest.mock import patch
import pandas as pd
import joblib

# --- 新增：强制 W&B 离线，防止测试时报错 ---
os.environ["WANDB_MODE"] = "offline" 
# ----------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import main

@pytest.fixture
def mock_training_env(tmp_path):
    # 模拟 SageMaker 目录结构
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # 创建模拟数据
    mock_data = pd.DataFrame({
        'Person ID': range(10),
        'Gender': ['Male', 'Female'] * 5,
        'Age': [30] * 10,
        'Occupation': ['Engineer'] * 10,
        'Sleep Duration': [7.0] * 10,
        'Quality of Sleep': [8] * 10,
        'Physical Activity Level': [50] * 10,
        'Stress Level': [5] * 10,
        'BMI Category': ['Normal'] * 10,
        'Blood Pressure': ['120/80'] * 10,
        'Heart Rate': [70] * 10,
        'Daily Steps': [5000] * 10,
        'Sleep Disorder': ['None', 'Sleep Apnea'] * 5
    })
    
    file_path = data_dir / "sleep_data.csv"
    mock_data.to_csv(file_path, index=False)
    return data_dir, model_dir

# --- 测试 1: Logistic Regression ---
def test_train_smoke_lr(mock_training_env):
    data_dir, model_dir = mock_training_env
    test_args = [
        'src/train.py', '--train', str(data_dir), '--model-dir', str(model_dir),
        '--model_type', 'logistic_regression', '--C', '0.1'
    ]
    with patch.object(sys, 'argv', test_args):
        main()
    assert (model_dir / "model.joblib").exists()

# --- 测试 2: SVM ---
def test_train_smoke_svm(mock_training_env):
    data_dir, model_dir = mock_training_env
    test_args = [
        'src/train.py', '--train', str(data_dir), '--model-dir', str(model_dir),
        '--model_type', 'svm', '--C', '0.5', '--kernel', 'linear'
    ]
    with patch.object(sys, 'argv', test_args):
        main()
    assert (model_dir / "model.joblib").exists()

# --- 测试 3: Random Forest ---
def test_train_smoke_rf(mock_training_env):
    data_dir, model_dir = mock_training_env
    test_args = [
        'src/train.py', '--train', str(data_dir), '--model-dir', str(model_dir),
        '--model_type', 'random_forest', '--n_estimators', '2'
    ]
    with patch.object(sys, 'argv', test_args):
        main()
    assert (model_dir / "model.joblib").exists()