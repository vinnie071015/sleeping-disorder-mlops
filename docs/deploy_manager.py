import yaml
import os
import datetime
import sys
import boto3 
from botocore.exceptions import ClientError

# -----------------------------------------------------------
# 路径定义
# -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'config.yaml')
ARCH_DOC_FILE = os.path.join(SCRIPT_DIR, 'architecture.md')

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------

def load_config():
    """读取 YAML 配置"""
    if not os.path.exists(CONFIG_FILE_PATH):
        raise FileNotFoundError(f"❌ 找不到配置文件: {CONFIG_FILE_PATH}")
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    """回写 YAML 配置 (更新状态)"""
    with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def update_architecture_status(config):
    """
    核心功能：将最新的部署状态覆盖写入 docs/architecture.md 中特定的标记区域。
    """
    status = config.get('deployment_status', {})
    last_run = status.get('last_run', 'N/A')
    
    # 状态图标映射
    icons = {
        "Pending": "⏳",
        "Success": "✅",
        "Failed": "❌",
        "Running": "🔄",
        "InService": "✅",         
        "Creating": "🔄",
        "Updating": "🔄",
        "Deleting": "🔄",
        "Unknown": "❓",
        "Not Configured": "⚙️",   
        "Deployed": "✅",          
        "Active": "✅",            
        "Failed (Not Found)": "❌"
    }

    sm_status = status.get('model_endpoint', 'Unknown')
    api_status = status.get('api_gateway', 'Unknown')
    lambda_status = status.get('lambda_function', 'Unknown')
    
    # 获取资源 ID/Name，用于 Details 列
    lambda_name = config.get('lambda', {}).get('function_name', 'N/A')
    api_id = config.get('api_gateway', {}).get('rest_api_id', 'N/A')
    
    # 生成 Markdown 表格内容 (新增 Lambda Function 行)
    status_content = f"""

_最后一次运行时间: {last_run}_

| 组件 (Component) | 状态 (Status) | 详情 (Details) |
| :--- | :--- | :--- |
| **S3 Storage** | {icons.get(status.get('s3_bucket', 'Unknown'), '❓')} {status.get('s3_bucket')} | Bucket: `{config['s3']['bucket_name']}` |
| **SageMaker Endpoint** | {icons.get(sm_status, '❓')} {sm_status} | Name: `{config['sagemaker']['endpoint_name']}` |
| **Lambda Function** | {icons.get(lambda_status, '❓')} {lambda_status} | Name: `{lambda_name}` |
| **API Gateway** | {icons.get(api_status, '❓')} {api_status} | ID: `{api_id}` |
| **Frontend App** | {icons.get(status.get('frontend', 'Pending'), '⏳')} {status.get('frontend')} | Local: `http://localhost:{config['frontend']['port']}` |

"""
    
    if not os.path.exists(ARCH_DOC_FILE):
        print(f"⚠️ 警告: 找不到文档 {ARCH_DOC_FILE}。请确保文件存在。", file=sys.stderr)
        return

    # 读取现有的文档
    with open(ARCH_DOC_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # === 修复：定义正确的标记字符串 ===
    start_marker = "<!-- DEPLOYMENT_STATUS_START -->"
    end_marker = "<!-- DEPLOYMENT_STATUS_END -->"

    start_index = content.find(start_marker)
    end_index = content.find(end_marker)

    if start_index != -1 and end_index != -1:
        # 计算替换的起点：在 start_marker 之后
        start_replace = start_index + len(start_marker)
        
        # 构建新的内容：[前缀] + [START_MARKER] + [新状态内容] + [END_MARKER] + [后缀]
        new_content = (
            content[:start_replace] + 
            status_content + 
            content[end_index:]
        )
        
        with open(ARCH_DOC_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"📄 [Docs] {ARCH_DOC_FILE} 已更新最新的部署状态 (覆盖写入成功)。")
    else:
        # 如果找不到标记，抛出 ValueError，确保脚本中断，而不是在 try/finally 中静默失败
        print(f"❌ 文档中缺少标记 {start_marker} 或 {end_marker}。请按要求修改文档。", file=sys.stderr)
        raise ValueError("Deployment markers not found in architecture.md") 


# ==========================================
# 核心状态检查函数 (Core Status Check Functions - Real Boto3 Calls)
# (这部分逻辑保持不变，用于收集详细诊断信息)
# ==========================================

def check_s3_bucket_status(config):
    """[REAL BOTO3] 检查 S3 桶是否存在且可访问"""
    bucket_name = config['s3']['bucket_name']
    region = config['project']['region'] if 'project' in config and 'region' in config['project'] else 'us-east-1'
    
    try:
        s3 = boto3.client('s3', region_name=region)
        s3.head_bucket(Bucket=bucket_name) 
        print(f"✅ S3 桶 '{bucket_name}' 状态: 存在且可访问.")
        return "Success"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return "Failed (404 Not Found)"
        elif error_code == '403':
            return "Failed (403 Forbidden)"
        else:
            print(f"❌ 错误: 检查 S3 桶时发生 AWS 错误 ({error_code}).")
            return f"Failed (AWS Error: {error_code})"
    except Exception as e:
        print(f"❌ 错误: 检查 S3 桶时发生本地错误: {e}")
        return "Failed (Local Error)"

def check_sagemaker_endpoint_status(config):
    """[REAL BOTO3] 检查 SageMaker Endpoint 的实时状态"""
    print("\n--- [Step 2] 检查 SageMaker Endpoint 状态 (Boto3) ---")
    endpoint_name = config['sagemaker']['endpoint_name']
    region = config['project']['region'] if 'project' in config and 'region' in config['project'] else 'us-east-1'
    
    try:
        sm = boto3.client('sagemaker', region_name=region)
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"✅ Endpoint '{endpoint_name}' 状态: {status}")
        return status
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if 'NotFoundException' in error_code:
            return "Failed (Not Found)"
        return f"Failed (AWS Error: {error_code})" 
    except Exception as e:
        print(f"❌ 错误: 检查 Endpoint 时发生本地错误: {e}")
        return "Failed (Local Error)"

def check_lambda_function_status(config):
    """[REAL BOTO3] 检查 Lambda 函数的实时状态"""
    print("\n--- [Step 3.1] 检查 Lambda 函数状态 (Boto3) ---")
    lambda_config = config.get('lambda', {})
    function_name = lambda_config.get('function_name')
    region = config['project']['region'] if 'project' in config and 'region' in config['project'] else 'us-east-1'

    if not function_name:
        print("❓ 警告: config.yaml 中缺少 Lambda 'function_name' 配置。")
        return "Not Configured"

    try:
        lambda_client = boto3.client('lambda', region_name=region)
        response = lambda_client.get_function(FunctionName=function_name)
        state = response['Configuration']['State'] 
        print(f"✅ Lambda 函数 '{function_name}' 状态: {state}")
        return state
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            return "Failed (Not Found)"
        return f"Failed (AWS Error: {error_code})"
    except Exception as e:
        print(f"❌ 错误: 检查 Lambda 时发生本地错误: {e}")
        return "Failed (Local Error)"

def check_api_gateway_status(config):
    """[REAL BOTO3] 检查 API Gateway RestApi 资源是否存在"""
    print("\n--- [Step 3.2] 检查 API Gateway 状态 (Boto3) ---")
    api_gateway_config = config.get('api_gateway', {})
    rest_api_id = api_gateway_config.get('rest_api_id')
    region = config['project']['region'] if 'project' in config and 'region' in config['project'] else 'us-east-1'

    if not rest_api_id:
        print("❓ 警告: config.yaml 中缺少 API Gateway 'rest_api_id' 配置。")
        return "Not Configured"

    try:
        apigateway = boto3.client('apigateway', region_name=region)
        apigateway.get_rest_api(restApiId=rest_api_id)
        
        print(f"✅ API Gateway ID '{rest_api_id}' 状态: Deployed.")
        return "Deployed"
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if 'NotFoundException' in str(e) or 'InvalidRestApiId' in str(e):
            return "Failed (Not Found)"
        return f"Failed (AWS Error: {error_code})"
    except Exception as e:
        print(f"❌ 错误: 检查 API Gateway 时发生本地错误: {e}")
        return "Failed (Local Error)"

# ==========================================
# 脚本主入口
# ==========================================

def main():
    print("🤖 部署管家 (Deployment Manager) 启动...")
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(str(e))
        return

    # 优化点：更新开始时间
    config['deployment_status']['last_run'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 将检查结果存储到变量中
    try:
        # 步骤 1: 检查 S3 桶
        config['deployment_status']['s3_bucket'] = check_s3_bucket_status(config)
        
        # 步骤 2: 检查 SageMaker Endpoint
        config['deployment_status']['model_endpoint'] = check_sagemaker_endpoint_status(config)
        
        # 步骤 3.1: 检查 Lambda Function
        config['deployment_status']['lambda_function'] = check_lambda_function_status(config)
        
        # 步骤 3.2: 检查 API Gateway 
        config['deployment_status']['api_gateway'] = check_api_gateway_status(config)
        
        # 步骤 4: 检查 Frontend 
        config['deployment_status']['frontend'] = "Pending"
        
        # === 核心优化点：在所有检查完成后，只更新文档一次 ===
        update_architecture_status(config)
        
        print("\n✨ 状态检查完成！请查看 docs/architecture.md 获取最新状态。")
        
    except Exception as e:
        # 捕获任何非 AWS 相关的中断错误（如文件标记缺失）
        print(f"\n❌ 部署管家中断: {e}")
        # 如果发生致命错误，也尝试将收集到的状态写入 config.yaml
    finally:
        save_config(config)

if __name__ == "__main__":
    main()