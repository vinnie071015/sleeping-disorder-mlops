import os
import sys
import boto3
import datetime
import traceback
import time

# ==========================================
# 核心组件：双向日志记录器 (DualLogger)
# ==========================================
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
        # 初始化时清空文件或创建新文件
        with open(self.log_file_path, "a", encoding='utf-8') as f:
            f.write(f"\n=== LOG SESSION STARTED: {datetime.datetime.now()} ===\n")

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

# ==========================================
# 辅助函数：上传 S3
# ==========================================
def upload_logs_to_s3(local_path, bucket_name):
    try:
        # 生成带时间戳的文件名，防止覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"debug_logs/full_log_capture_{timestamp}.txt"
        
        # 使用原生 stdout 打印，防止递归干扰
        sys.__stdout__.write(f"\n[S3 Upload] Uploading {local_path} to s3://{bucket_name}/{s3_key} ...\n")
        
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket_name, s3_key)
        
        sys.__stdout__.write(f"✅ [S3 Upload] Success! Log is safe.\n")
    except Exception as e:
        sys.__stdout__.write(f"❌ [S3 Upload] Failed: {e}\n")

# ==========================================
# 主流程 (模拟各种打印情况)
# ==========================================
if __name__ == '__main__':
    # 1. 定义日志文件路径 (放在 /tmp 最安全)
    LOG_FILE = "/tmp/captured_log.txt"
    BUCKET_NAME = 'sleep-disorder-mlops-bucket' # 你的桶名

    # 2. 劫持标准输出和错误输出
    sys.stdout = DualLogger(sys.stdout, LOG_FILE)
    sys.stderr = DualLogger(sys.stderr, LOG_FILE)

    print("--- 🚀 SCRIPT START ---")
    
    try:
        # 3. 模拟正常信息
        print(f"INFO: Current working directory: {os.getcwd()}")
        print("INFO: Loading modules...")
        time.sleep(1)
        print("INFO: Data processing...")
        
        # 4. 模拟一个警告
        print("⚠️ WARNING: This is a simulated warning message.")
        
        # 5. 模拟一个致命错误 (除以零)
        print("INFO: Attempting risky calculation...")
        result = 1 / 0 

    except Exception:
        # 6. 捕获报错堆栈 (这部分最重要，看能不能被写入文件)
        print("\n❌ CRASH DETECTED! Printing Traceback:")
        traceback.print_exc()
        
    finally:
        # 7. 无论上面是否报错，这里都会执行
        print("\n--- 🏁 SCRIPT FINISHING ---")
        print("INFO: Initiating log upload procedure...")
        
        # 恢复标准输出，确保 boto3 不受干扰
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        upload_logs_to_s3(LOG_FILE, BUCKET_NAME)
