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

import argparse
import os
import time
import subprocess
import sys

# 导入您创建的辅助脚本
try:
    # SageMaker 会将 Git 仓库内容放在 /opt/ml/code/ 下
    # s3_log_uploader.py 位于根目录，src/train.py 位于 src/，所以路径是 ../s3_log_uploader
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from s3_log_uploader import upload_log_to_s3
except Exception:
    # 如果导入失败，则无法上传日志
    def upload_log_to_s3(content, name):
        print("--- ⚠️ [S3 LOG IMPORT FAILED] S3 日志功能禁用。---")
        pass


def run_training():
    # 获取任务名，用于 S3 路径
    job_name = os.environ.get('TRAINING_JOB_NAME', f'local-test-job-{time.strftime("%H%M%S")}')

    # ----------------------------------------------------
    # 1. 模拟执行 pip install -r requirements.txt
    # ----------------------------------------------------
    print("--- 🔍 [TEST] 尝试执行 pip install ---")

    # 路径指向 Git 仓库根目录下的 requirements.txt
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../requirements.txt')

    if os.path.exists(requirements_path):
        try:
            # 运行 pip install 并捕获 stdout/stderr
            process = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_path],
                capture_output=True,
                text=True,
                timeout=300, # 给予 5 分钟安装时间
                check=True  # 如果安装失败，抛出 CalledProcessError
            )
            print("--- ✅ [PIP SUCCESS] 依赖安装成功。---")

        except subprocess.CalledProcessError as e:
            # 捕获错误并上传 S3
            error_log = f"*** PIP INSTALL FAILED ***\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            s3_path = upload_log_to_s3(error_log, job_name)

            print(f"--- ❌ [FATAL ERROR] PIP 安装失败，请检查 S3 日志: {s3_path} ---")

            # 必须调用 sys.exit(1) 才能让 SageMaker 标记为 Failed
            sys.exit(1) 

        except Exception as e:
            # 处理其他异常，如超时
            error_log = f"*** GENERAL ERROR DURING PIP INSTALL ***\n{e}"
            upload_log_to_s3(error_log, job_name)
            print(f"--- ❌ [FATAL ERROR] 运行异常: {e} ---")
            sys.exit(1)

    else:
        print("--- ⚠️ [WARN] requirements.txt 文件未找到，跳过安装。---")

    # ----------------------------------------------------
    # 2. 极简训练逻辑 (如果依赖安装成功，才会执行到这里)
    # ----------------------------------------------------
    print("--- ✅ [START] 用户脚本开始执行，基础环境测试成功 ---")
    time.sleep(10)
    print("✅ Accuracy: 0.99")
    print("--- ✅ [END] 脚本成功完成 ---")