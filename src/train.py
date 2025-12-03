import os
import sys
import argparse
import datetime
import traceback
import subprocess
import time

# ==========================================
# 0. Core Configuration and Logging Component
# ==========================================

# ⚠️ Please confirm your S3 bucket name
LOG_BUCKET_NAME = 'sleep-disorder-mlops-bucket' 
LOG_FILE_PATH = "/tmp/captured_log.txt"

class DualLogger:
    """
    Interception sys.stdout and sys.stderr,
    Directing content simultaneously to:
    1. Console (CloudWatch)
    2. Local file (/tmp/log.txt) -> For S3 upload
    """
    def __init__(self, original_stream, log_file_path):
        self.terminal = original_stream
        self.log_file_path = log_file_path
        # In append mode for continuous logging
    
    def write(self, message):
        # 1. Print to console as usual
        self.terminal.write(message)
        # 2. Append to file
        try:
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(message)
        except Exception as e:
            pass 

    def flush(self):
        self.terminal.flush()

def upload_logs_to_s3(local_path, bucket_name):
    """Attempts to upload the log file to S3"""
    try:
        import boto3 # Lazy import to ensure boto3 is available
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"debug_logs/train_failure_log_{timestamp}.txt"
        
        # Use native stdout to avoid recursive loop interference
        sys.__stdout__.write(f"\n[S3 Upload] Uploading logs to s3://{bucket_name}/{s3_key} ...\n")
        
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket_name, s3_key)
        
        sys.__stdout__.write(f"✅ [S3 Upload] Success! Log saved to S3.\n")
    except Exception as e:
        sys.__stdout__.write(f"❌ [S3 Upload] Failed: {e}\n")

# ==========================================
# 1. Dependency Installation (Nuclear Cleanup Mode)
# ==========================================
def install_dependencies():
    print("\n📦 [INIT] Start environment cleanup & installation...", flush=True)
    
    # --- 1. Purge Pre-installed Libraries (Nuclear Cleanup) ---
    troublemakers = ["numpy", "pandas", "scikit-learn", "joblib"]
    print(f"   --- Purging pre-installed libraries: {troublemakers}...", flush=True)
    try:
        # -y automatically confirms uninstallation
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y"] + troublemakers)
        print("   ✅ Cleanup complete.", flush=True)
    except Exception as e:
        # Non-fatal if uninstallation fails (e.g., package wasn't there)
        print(f"   ⚠️ Cleanup warning (non-fatal): {e}", flush=True)

    # --- 2. Upgrade pip ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except:
        pass

    # --- 3. Reinstall specific versions ---
    packages = [
        "numpy==1.26.4",  # Specific stable version
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
        "matplotlib",
        "seaborn",
        "joblib",
        "wandb"
    ]
    
    for package in packages:
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            print(f"   - Installing new: {package} ...", flush=True)
            subprocess.check_call(cmd)
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to install {package}. Error: {e}", flush=True)
            
    print("✅ [INIT] Fresh dependencies installed.\n", flush=True)

# ==========================================
# 2. Training Logic (Encapsulated)
# ==========================================
def perform_training(args):
    print("🔄 [IMPORT] Loading ML libraries...", flush=True)
    
    # Imports must be inside the function to run AFTER install_dependencies
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
    
    # [W&B ADDITION] Try to import wandb safely
    wandb_available = False
    try:
        import wandb
        wandb_available = True
        print("✅ [W&B] Library imported successfully.", flush=True)
    except ImportError:
        print("⚠️ [W&B] Library not found. Skipping W&B logging.", flush=True)

    # Dynamic import of src
    sys.path.append(os.getcwd())
    try:
        from src.data_processor import load_data, clean_data
        print("✅ [IMPORT] src.data_processor loaded.", flush=True)
    except ImportError as e:
        print(f"❌ [IMPORT] Failed to import src.data_processor: {e}", flush=True)
    
    # --------------------------------------------------------
    # Helper Functions (Internal Definitions)
    # --------------------------------------------------------
    def get_model(model_args):
        # [FIX] Robust String Cleaning: Remove extra quotes and whitespace
        model_type = model_args.model_type.strip().replace('"', '').lower()
        
        # We must check against the long names used in your Notebook Hyperparameter config
        if model_type == 'logistic_regression': 
            print("STATUS: Selected Logistic Regression model.", flush=True)
            return LogisticRegression(C=model_args.C)
        elif model_type == 'svm': 
            print("STATUS: Selected SVM model.", flush=True)
            return SVC(C=model_args.C, kernel=model_args.kernel)
        elif model_type == 'random_forest': 
            print("STATUS: Selected Random Forest model.", flush=True)
            return RandomForestClassifier(n_estimators=model_args.n_estimators)
        else: 
            raise ValueError(f"Unknown model type: {model_args.model_type} (Cleaned to: {model_type})")

    def create_pipeline(cat_cols, num_cols, m_args):
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        model = get_model(m_args)
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # --------------------------------------------------------
    # [W&B ADDITION] Initialize W&B Run
    # --------------------------------------------------------
    if wandb_available:
        try:
            # Clean model type for cleaner charts
            clean_model_type = args.model_type.strip().replace('"', '')
            
            wandb.init(
                project="sleep-disorder-mlops", # Replace with your project name if needed
                config={
                    "model_type": clean_model_type,
                    "n_estimators": args.n_estimators,
                    "C": args.C,
                    "kernel": args.kernel,
                    "full_args": vars(args)
                }
            )
            print("✅ [W&B] Run initialized successfully.", flush=True)
        except Exception as e:
            print(f"⚠️ [W&B] Initialization failed (check API Key?): {e}", flush=True)
            wandb_available = False # Disable subsequent logging if init fails

    # --------------------------------------------------------
    # Business Logic Start
    # --------------------------------------------------------
    print("\n--- 1. Data Loading ---", flush=True)
    
    data_dir = args.train
    print(f"DATA_DIAG: Target data directory: {data_dir}", flush=True)

    if not data_dir:
        raise ValueError("❌ Error: args.train is Empty! Check argparse defaults.")

    file_path = os.path.join(data_dir, "sleep_data.csv")
    print(f"DATA_DIAG: Full file path: {file_path}", flush=True)

    if not os.path.exists(file_path):
        print(f"❌ [ERROR] File not found at {file_path}. Listing dir contents:", flush=True)
        if os.path.exists(data_dir):
            print(os.listdir(data_dir), flush=True)
        else:
            print(f"   Directory {data_dir} does not exist!", flush=True)
        raise FileNotFoundError(f"Data file missing: {file_path}")

    df = load_data(file_path)
    df = clean_data(df)
    print(f"DATA_DIAG: Data Loaded. Shape: {df.shape}", flush=True)

    # Feature Engineering
    target_col = 'sleep_disorder'
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} missing.")

    df[target_col] = df[target_col].fillna('None')
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    X = df.drop(columns=[target_col, 'person_id'], errors='ignore')
    y = df[target_col]
    
    # Training
    print("\n--- 2. Training ---", flush=True)
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['number']).columns
    
    pipeline = create_pipeline(cat_features, num_features, args)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    print("STATUS: Model fitting completed.", flush=True)

    # Evaluation and Saving
    print("\n--- 3. Evaluation & Saving ---", flush=True)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    
    # [W&B ADDITION] Log Metrics
    if wandb_available:
        try:
            wandb.log({"accuracy": acc, "test_accuracy": acc})
            print(f"✅ [W&B] Logged accuracy: {acc:.4f}", flush=True)
        except Exception as e:
            print(f"⚠️ [W&B] Logging failed: {e}", flush=True)

    # Using the exact format for SageMaker metric capture
    print(f"✅ Accuracy: {acc:.4f}", flush=True) 

    # Saving
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.joblib"))
    print(f"✅ FINAL: Model saved to {args.model_dir}", flush=True)
    
    # [W&B ADDITION] Finish Run
    if wandb_available:
        wandb.finish()


# ==========================================
# 3. Main Entry Point (with Logging Hijacking)
# ==========================================
if __name__ == '__main__':
    # 1. Initialize Log File
    with open(LOG_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(f"=== TRAINING SESSION STARTED: {datetime.datetime.now()} ===\n")

    # 2. Hijack Output Streams
    sys.stdout = DualLogger(sys.stdout, LOG_FILE_PATH)
    sys.stderr = DualLogger(sys.stderr, LOG_FILE_PATH)

    print("--- 🚀 SCRIPT START ---", flush=True)
    
    try:
        # 3. Argument Parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', type=str, default='svm')
        parser.add_argument('--n_estimators', type=int, default=100)
        parser.add_argument('--C', type=float, default=1.0)
        parser.add_argument('--kernel', type=str, default='rbf')
        
        # Robust Path Handling
        env_sm_channel = os.environ.get('SM_CHANNEL_TRAINING')
        default_data_path = env_sm_channel if env_sm_channel else '/opt/ml/input/data/train'
        
        parser.add_argument('--train', type=str, default=default_data_path)
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
        
        args, _ = parser.parse_known_args() # Use parse_known_args for better compatibility with SageMaker
        
        print(f"INFO: Arguments: {args}", flush=True)
        print(f"INFO: Env SM_CHANNEL_TRAINING: {env_sm_channel}", flush=True)
        print(f"INFO: Effective Data Path: {args.train}", flush=True)

        # 4. Execute Installation and Training
        install_dependencies()
        perform_training(args)

    except Exception:
        # 5. Catch all crashes and print traceback
        print("\n❌ CRASH DETECTED! Printing Traceback:", flush=True)
        traceback.print_exc()
        
    finally:
        # 6. Final Log Upload
        print("\n--- 🏁 SCRIPT FINISHING ---", flush=True)
        print("INFO: Initiating log upload procedure...", flush=True)
        
        # Restore standard output streams before calling boto3
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        upload_logs_to_s3(LOG_FILE_PATH, LOG_BUCKET_NAME)