"""
SageMaker training script supporting 3 Course Models: LR, SVM, RF.
"""
import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Week 2
from sklearn.svm import SVC                          # Week 3
from sklearn.ensemble import RandomForestClassifier  # Week 4
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 尝试导入 wandb
try:
    import wandb
except ImportError:
    wandb = None

from src.data_processor import load_data, clean_data

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker 路径
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # --- 模型选择 (LR, SVM, RF) ---
    parser.add_argument('--model_type', type=str, default='random_forest', 
                        choices=['logistic_regression', 'svm', 'random_forest'])

    # --- 超参数 ---
    # 1. 通用/SVM/LR 参数
    parser.add_argument('--C', type=float, default=1.0) # 正则化强度
    
    # 2. SVM 专属
    parser.add_argument('--kernel', type=str, default='rbf')
    
    # 3. Random Forest 专属
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)

    return parser.parse_args()

def get_model(args):
    """工厂模式：根据参数返回对应的模型"""
    print(f"🏗️ Building model: {args.model_type.upper()}...")
    
    if args.model_type == 'logistic_regression':
        # Week 2: Logistic Regression (需要 max_iter 以防不收敛)
        return LogisticRegression(C=args.C, solver='lbfgs', max_iter=1000, random_state=42)
    
    elif args.model_type == 'svm':
        # Week 3: SVM
        return SVC(C=args.C, kernel=args.kernel, probability=True, random_state=42)
    
    elif args.model_type == 'random_forest':
        # Week 4: Random Forest
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)

def save_plot_confusion_matrix(y_test, y_pred, output_dir):
    """保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    if wandb:
        wandb.log({"confusion_matrix": wandb.Image(save_path)})

def perform_bias_audit(X_test_raw, y_test, y_pred):
    """执行偏见审计 (Bias Audit)"""
    print("\n--- ⚖️ Bias Audit Report ---")
    audit_df = X_test_raw.copy()
    audit_df['Actual'] = y_test.values
    audit_df['Predicted'] = y_pred
    
    # 兼容列名大小写
    gender_col = 'gender' if 'gender' in audit_df.columns else 'Gender'
    
    if gender_col in audit_df.columns:
        for name, group in audit_df.groupby(gender_col):
            acc = accuracy_score(group['Actual'], group['Predicted'])
            print(f"Group '{name}': Accuracy = {acc:.4f} (Count: {len(group)})")
            if wandb:
                wandb.log({f"bias_acc_{name}": acc})
    else:
        print("Warning: Gender column not found for bias audit.")

def main():
    args = parse_args()
    if wandb:
        wandb.init(project="sleep-disorder-mlops", job_type="train", config=args)

    print(f"--- 🚀 Training Started ({args.model_type}) ---")

    # 1. Load & Clean
    file_path = os.path.join(args.train, "sleep_data.csv")
    df = load_data(file_path)
    df = clean_data(df)

    # 2. Prepare Target
    target_col = 'sleep_disorder'
    df[target_col] = df[target_col].fillna('None')
    
    # Label Encode Target
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    X = df.drop(columns=[target_col, 'person_id'])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Build Pipeline
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            # LR 和 SVM 都强烈依赖 Scaling
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    model = get_model(args)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    if wandb:
        wandb.log({"accuracy": accuracy, "f1_score": f1})

    # 6. Audits & Plots
    save_plot_confusion_matrix(y_test, y_pred, args.model_dir)
    perform_bias_audit(X_test, y_test, y_pred)

    # 7. Save Model
    model_output_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_output_path)
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.joblib"))
    print(f"💾 Model saved to: {args.model_dir}")

if __name__ == '__main__':
    main()