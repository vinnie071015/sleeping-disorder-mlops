import os
import sys
import argparse
import datetime
import traceback
import subprocess
import time

# ==========================================
# 0. 核心配置与日志组件
# ==========================================

# ⚠️ 请确认你的 S3 桶名称
LOG_BUCKET_NAME = 'sleep-disorder-mlops-bucket' 
LOG_FILE_PATH = "/tmp/captured_log.txt"

class DualLogger:
    """
    拦截 sys.stdout 和 sys.stderr，
    将内容同时输出到：
    1. 控制台 (CloudWatch)
    2. 本地文件 (/tmp/log.txt) -> 用于上传 S3
    """
    def __init__(self, original_stream, log_file_path):
        self.terminal = original_stream
        self.log_file_path = log_file_path
        # 初始化时，如果是 stdout 则不需要清空（避免双重清空），这里简单处理：追加模式
        # 实际由外部控制文件初始化
        
    def write(self, message):
        # 1. 照常打印到控制台
        self.terminal.write(message)
        # 2. 追加写入文件
        try:
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(message)
        except Exception:
            pass 

    def flush(self):
        self.terminal.flush()

def upload_logs_to_s3(local_path, bucket_name):
    """尝试将日志文件上传到 S3"""
    try:
        import boto3 # 延迟导入，确保 boto3 可用
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"debug_logs/train_failure_log_{timestamp}.txt"
        
        # 使用原生 stdout 打印，防止递归死循环
        sys.__stdout__.write(f"\n[S3 Upload] Uploading logs to s3://{bucket_name}/{s3_key} ...\n")
        
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket_name, s3_key)
        
        sys.__stdout__.write(f"✅ [S3 Upload] Success! Log saved to S3.\n")
    except Exception as e:
        sys.__stdout__.write(f"❌ [S3 Upload] Failed: {e}\n")

# ==========================================
# 1. 依赖安装 (在导入 ML 库之前执行)
# ==========================================
def install_dependencies():
    print("\n📦 [INIT] Start installing dependencies...", flush=True)
    packages = [
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "wandb"
    ]
    for package in packages:
        try:
            print(f"   - Installing {package}...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to install {package}. Error: {e}", flush=True)
    print("✅ [INIT] Dependencies installed.\n", flush=True)

# ==========================================
# 2. 训练逻辑 (封装在函数中，避免全局导入报错)
# ==========================================
def perform_training(args):
    print("🔄 [IMPORT] Loading ML libraries...", flush=True)
    
    # --- 这里的 Import 必须放在函数内部 ---
    # 因为在 main() 运行 install_dependencies() 之前，这些包可能不存在
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    
    # 动态导入 src
    sys.path.append(os.getcwd())
    try:
        from src.data_processor import load_data, clean_data
        print("✅ [IMPORT] src.data_processor loaded.", flush=True)
    except ImportError as e:
        print(f"❌ [IMPORT] Failed to import src.data_processor: {e}", flush=True)
        # 继续尝试运行，或者在这里 raise
    
    # --------------------------------------------------------
    # Helper Functions (内部定义)
    # --------------------------------------------------------
    def get_model(model_args):
        if model_args.model_type == 'lr': return LogisticRegression(C=model_args.C)
        elif model_args.model_type == 'svm': return SVC(C=model_args.C, kernel=model_args.kernel)
        elif model_args.model_type == 'rf': return RandomForestClassifier(n_estimators=model_args.n_estimators)
        else: raise ValueError(f"Unknown model type: {model_args.model_type}")

    def create_pipeline(cat_cols, num_cols, m_args):
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        model = get_model(m_args)
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # --------------------------------------------------------
    # 业务逻辑 Start
    # --------------------------------------------------------
    print("\n--- 1. Data Loading ---", flush=True)
    
    # 如果本地测试没有 path，提供默认值防止报错
    data_dir = args.train if args.train else "./data" 
    file_path = os.path.join(data_dir, "sleep_data.csv")
    print(f"DATA_DIAG: Loading from {file_path}", flush=True)

    if not os.path.exists(file_path):
        print(f"❌ [ERROR] File not found at {file_path}. Listing dir:", flush=True)
        if os.path.exists(data_dir):
            print(os.listdir(data_dir), flush=True)
        raise FileNotFoundError(f"Data file missing: {file_path}")

    df = load_data(file_path)
    df = clean_data(df)
    print(f"DATA_DIAG: Data Loaded. Shape: {df.shape}", flush=True)

    # 特征处理
    target_col = 'sleep_disorder'
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} missing.")

    df[target_col] = df[target_col].fillna('None')
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    X = df.drop(columns=[target_col, 'person_id'], errors='ignore')
    y = df[target_col]
    
    # 训练
    print("\n--- 2. Training ---", flush=True)
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['number']).columns
    
    pipeline = create_pipeline(cat_features, num_features, args)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    print("STATUS: Model fitting completed.", flush=True)

    # 评估与保存
    print("\n--- 3. Evaluation & Saving ---", flush=True)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"RESULT: Accuracy: {acc:.4f}", flush=True)

    # 保存
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.joblib"))
    print(f"✅ FINAL: Model saved to {args.model_dir}", flush=True)


# ==========================================
# 3. 主入口 (包含日志劫持)
# ==========================================
if __name__ == '__main__':
    # 1. 初始化日志文件
    with open(LOG_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(f"=== TRAINING SESSION STARTED: {datetime.datetime.now()} ===\n")

    # 2. 劫持输出
    sys.stdout = DualLogger(sys.stdout, LOG_FILE_PATH)
    sys.stderr = DualLogger(sys.stderr, LOG_FILE_PATH)

    print("--- 🚀 SCRIPT START ---", flush=True)
    
    try:
        # 3. 解析参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', type=str, default='svm')
        parser.add_argument('--n_estimators', type=int, default=100)
        parser.add_argument('--C', type=float, default=1.0)
        parser.add_argument('--kernel', type=str, default='rbf')
        # SageMaker 环境变量默认值
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/tmp/model'))
        
        args, _ = parser.parse_known_args() # 使用 parse_known_args 容错性更好

        print(f"INFO: Arguments: {args}", flush=True)
        print(f"INFO: Env SM_CHANNEL_TRAINING: {os.environ.get('SM_CHANNEL_TRAINING')}", flush=True)

        # 4. 执行安装和训练
        install_dependencies()
        perform_training(args)

    except Exception:
        # 5. 捕获一切崩溃
        print("\n❌ CRASH DETECTED! Printing Traceback:", flush=True)
        traceback.print_exc()
        # 此时 sys.stderr 也是 DualLogger，所以 traceback 也会写入文件
        
    finally:
        # 6. 最终上传日志
        print("\n--- 🏁 SCRIPT FINISHING ---", flush=True)
        print("INFO: Initiating log upload procedure...", flush=True)
        
        # 恢复标准输出，确保 boto3 不受干扰
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        upload_logs_to_s3(LOG_FILE_PATH, LOG_BUCKET_NAME)