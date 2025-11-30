import boto3
import os
import sys

# --- 修改配置区域 ---
# 更新路径：指向 data/ 文件夹
LOCAL_FILE_PATH = 'data/Sleep_health_and_lifestyle_dataset.csv' 

# S3 配置 (保持不变)
BUCKET_NAME = 'sleep-disorder-mlops-bucket' 
S3_KEY_PREFIX = 'raw_data/'
S3_FILE_NAME = 'sleep_data.csv' 

def upload_to_s3():
    # 1. 检查本地文件是否存在
    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"❌ 错误: 找不到本地文件 '{LOCAL_FILE_PATH}'")
        # 打印当前工作目录，方便排查
        print(f"   当前工作目录: {os.getcwd()}")
        sys.exit(1)

    # 2. 初始化 S3 客户端
    s3 = boto3.client('s3')
    
    s3_full_key = f"{S3_KEY_PREFIX}{S3_FILE_NAME}"

    try:
        print(f"⏳ 正在将 '{LOCAL_FILE_PATH}' 上传至 s3://{BUCKET_NAME}/{s3_full_key} ...")
        
        s3.upload_file(LOCAL_FILE_PATH, BUCKET_NAME, s3_full_key)
        
        print("✅ 上传成功！")
        print(f"   S3 URI: s3://{BUCKET_NAME}/{s3_full_key}")
        print("   这是您的 'Source of Truth' (单一真实数据源)。")
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_to_s3()