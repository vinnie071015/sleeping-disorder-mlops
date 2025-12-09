import boto3
import os
import sys

# --- Configuration area modification ---
# Update path: pointing to data/ folder
LOCAL_FILE_PATH = 'data/Sleep_health_and_lifestyle_dataset.csv' 

# S3 Configuration (keep unchanged)
BUCKET_NAME = 'sleep-disorder-mlops-bucket' 
S3_KEY_PREFIX = 'raw_data/'
S3_FILE_NAME = 'sleep_data.csv' 

def upload_to_s3():
    # 1. Check if local file exists
    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"❌ Error: Cannot find local file '{LOCAL_FILE_PATH}'")
        # Print current working directory for troubleshooting
        print(f"   Current working directory: {os.getcwd()}")
        sys.exit(1)

    # 2. Initialize S3 client
    s3 = boto3.client('s3')
    
    s3_full_key = f"{S3_KEY_PREFIX}{S3_FILE_NAME}"

    try:
        print(f"⏳ Uploading '{LOCAL_FILE_PATH}' to s3://{BUCKET_NAME}/{s3_full_key} ...")
        
        s3.upload_file(LOCAL_FILE_PATH, BUCKET_NAME, s3_full_key)
        
        print("✅ Upload successful!")
        print(f"   S3 URI: s3://{BUCKET_NAME}/{s3_full_key}")
        print("   This is your 'Source of Truth' (Single Source of Truth for Data).")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_to_s3()