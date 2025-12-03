# """
# SageMaker training script supporting 3 Course Models: LR, SVM, RF.
# Includes verbose debugging and environment checks.
# """
# import argparse
# import os
# import sys
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# try:
#     import wandb
# except ImportError:
#     wandb = None

# from src.data_processor import load_data, clean_data # <--- 路径修复后的导入

# # ... (parse_args, get_model, save_plot_confusion_matrix, perform_bias_audit 函数保持不变) ...

# def create_pipeline(categorical_features, numerical_features, n_estimators=None, C=None, kernel=None, model_type=None):
#     """
#     构建 Scikit-learn Pipeline。
#     (重构：将 Pipeline 逻辑移到函数中，方便主函数瘦身)
#     """
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ]
#     )

#     model = get_model(argparse.Namespace(
#         model_type=model_type, n_estimators=n_estimators, C=C, kernel=kernel))
        
#     return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


# def main():
#     """主训练流程 (包含了详细的步骤打印)"""
#     args = parse_args()
#     # --------------------------------------------------------
#     print("\n✅ SCRIPT START: SageMaker 容器环境已就绪，开始执行 train.py ...") # 👈 增加这个明确的信号
#     # --------------------------------------------------------
#     print("\n--- 1. 环境诊断与参数接收 (Receiving Instructions) ---")
    
#     # 打印关键 SageMaker 环境变量
#     print(f"ENV_DIAG: SM_CHANNEL_TRAINING = {os.environ.get('SM_CHANNEL_TRAINING')}")
#     print(f"ENV_DIAG: SM_MODEL_DIR = {os.environ.get('SM_MODEL_DIR')}")
    
#     # 修复 Python 模块导入路径 (重复执行确保安全)
#     sys.path.append(os.getcwd()) 
    
#     print(f"PARAM_DIAG: Model Type: {args.model_type}, N_Estimators: {args.n_estimators}, C: {args.C}")
#     print("--------------------------------------------------------")
    
    
#     # --- 2. 数据加载与清洗 ---
    
#     # 确认数据在容器内的实际路径
#     data_dir_path = args.train
#     file_path = os.path.join(data_dir_path, "sleep_data.csv")
    
#     print(f"\n--- 2. Data Loading ---")
#     print(f"DATA_DIAG: Attempting to load file from: {file_path}")
    
#     try:
#         df = load_data(file_path)
#         df = clean_data(df)
#     except Exception as e:
#         # 如果加载或清洗失败，打印自定义错误并退出
#         print(f"❌ FATAL ERROR: Data loading/cleaning failed at runtime: {e}")
#         sys.exit(1) # 强制退出，避免继续运行
        

#     # --- 3. 特征和目标准备 ---
#     target_col = 'sleep_disorder'
#     df[target_col] = df[target_col].fillna('None')
    
#     le = LabelEncoder()
#     df[target_col] = le.fit_transform(df[target_col])
    
#     X = df.drop(columns=[target_col, 'person_id'])
#     y = df[target_col]
    
#     print(f"DATA_DIAG: Final Feature Count: {len(X.columns)}")
#     print(f"DATA_DIAG: Target Classes: {le.classes_}")
    
#     # --- 4. 训练与评估 ---
#     print(f"\n--- 4. Model Training ---")
    
#     cat_features = X.select_dtypes(include=['object']).columns
#     num_features = X.select_dtypes(include=['number']).columns

#     # 组装 Pipeline (使用重构后的 create_pipeline 函数)
#     pipeline = create_pipeline(
#         cat_features, num_features, args.n_estimators, args.C, args.kernel, args.model_type
#     )

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     try:
#         print("STATUS: Fitting model to data...")
#         pipeline.fit(X_train, y_train)
#         print("STATUS: Model fitting completed.")
#     except Exception as e:
#         print(f"❌ FATAL ERROR: Model fitting crashed: {e}")
#         sys.exit(1)


#     # --- 5. 产物生成与保存 ---
#     print("\n--- 5. Artifacts Generation ---")
#     y_pred = pipeline.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     print(f"RESULT: Accuracy: {accuracy:.4f}")
    
#     # Audits & Plots
#     save_plot_confusion_matrix(y_test, y_pred, args.model_dir)
#     perform_bias_audit(X_test, y_test, y_pred)
    
#     # Save Model
#     model_output_path = os.path.join(args.model_dir, "model.joblib")
#     joblib.dump(pipeline, model_output_path)
#     joblib.dump(le, os.path.join(args.model_dir, "label_encoder.joblib"))
#     print(f"✅ FINAL STATUS: Model saved successfully to {args.model_dir}")


# if __name__ == '__main__':
#     main()

import os
import sys
import subprocess
import time
import boto3

# --- 配置部分 ---
# 这里定义我们要“手动”安装的高风险库
# 强烈建议锁定版本，以避免我们之前推测的兼容性问题
RISKY_PACKAGES = [
    "numpy==1.23.5",      # 锁定旧版本以兼容 SageMaker SKLearn 容器
    "pandas==1.5.3",      # 锁定 1.x 版本
    "scikit-learn==1.2.2" # 与容器版本匹配
]

# 获取任务名和区域
JOB_NAME = os.environ.get('TRAINING_JOB_NAME', f'debug-job-{int(time.time())}')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
# 尝试从环境变量获取 Bucket，如果没有则硬编码您的 Bucket
BUCKET_NAME = 'sleep-disorder-mlops-bucket' 

def upload_log_to_s3(content, filename_suffix):
    """上传日志到 S3 的辅助函数"""
    try:
        s3 = boto3.client('s3', region_name=REGION)
        s3_key = f'sagemaker-logs/manual-install-debug/{JOB_NAME}/{filename_suffix}.txt'
        s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=content.encode('utf-8'))
        print(f"--- ✅ [S3 UPLOAD] 日志已上传: s3://{BUCKET_NAME}/{s3_key} ---")
        return f"s3://{BUCKET_NAME}/{s3_key}"
    except Exception as e:
        print(f"--- ❌ [S3 ERROR] 上传失败: {e} ---")
        return None

def install_risky_packages():
    """在脚本内部手动运行 pip install"""
    print(f"--- 🛠️ [INSTALL] 开始手动安装库: {RISKY_PACKAGES} ---")
    
    cmd = [sys.executable, "-m", "pip", "install"] + RISKY_PACKAGES
    
    # 执行命令并捕获所有输出
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # 拼接完整日志
    full_log = (
        f"COMMAND: {' '.join(cmd)}\n"
        f"RETURN CODE: {result.returncode}\n\n"
        f"====== STDOUT ======\n{result.stdout}\n\n"
        f"====== STDERR ======\n{result.stderr}\n"
    )
    
    # 无论成功失败，都上传日志
    if result.returncode == 0:
        print("--- ✅ [INSTALL SUCCESS] 手动安装成功！---")
        upload_log_to_s3(full_log, "install_success_log")
        return True
    else:
        print("--- ❌ [INSTALL FAILED] 手动安装失败！---")
        print(result.stderr[-500:]) # 打印最后500字符到控制台(如果有的话)
        s3_path = upload_log_to_s3(full_log, "install_failure_log")
        print(f"详细错误日志请查看 S3: {s3_path}")
        return False

if __name__ == "__main__":
    print("--- 🚀 [START] User script started. Safe dependencies loaded. ---")
    
    # 1. 尝试安装高风险库
    success = install_risky_packages()
    
    if not success:
        print("--- 💀 [ABORT] 核心库安装失败，脚本退出。 ---")
        # 退出码 1 让 SageMaker 知道任务失败了
        sys.exit(1)
        
    # 2. 如果安装成功，尝试导入测试
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print(f"--- ✅ [IMPORT TEST] Libraries imported successfully.")
        print(f"Numpy: {np.__version__}, Pandas: {pd.__version__}, Sklearn: {sklearn.__version__}")
        
        # 上传一个最终的成功标志
        upload_log_to_s3("All systems go! Environment is ready.", "final_success")
        
    except ImportError as e:
        error_msg = f"Install reported success, but IMPORT failed: {e}"
        print(error_msg)
        upload_log_to_s3(error_msg, "import_error_log")
        sys.exit(1)

    # 3. 模拟极简训练
    print("--- ⏳ [TRAINING] Simulating training loop... ---")
    time.sleep(5)
    print("✅ Accuracy: 0.99")
    print("--- ✅ [DONE] Script finished. ---")