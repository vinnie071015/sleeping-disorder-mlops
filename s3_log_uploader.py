import boto3
import os

def upload_log_to_s3(log_content, job_name):
    """将日志内容上传到 S3 桶，以便在任务失败时进行诊断。"""
    try:
        # 假设您的数据桶可以用于存储日志
        bucket = 'sleep-disorder-mlops-bucket' 
        region = os.environ.get('AWS_REGION', 'us-east-1')

        s3 = boto3.client('s3', region_name=region)
        s3_key = f'sagemaker-logs/pip-install-error/{job_name}/log.txt'

        s3.put_object(Bucket=bucket, Key=s3_key, Body=log_content.encode('utf-8'))

        print(f"--- ✅ [LOG UPLOADED] 诊断日志已上传到 s3://{bucket}/{s3_key} ---")
        return f"s3://{bucket}/{s3_key}"

    except Exception as e:
        print(f"--- ❌ [LOG UPLOAD FAILED] 无法上传 S3 日志: {e} ---")
        return None