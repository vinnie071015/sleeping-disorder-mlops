import yaml
import os
import datetime
import time
import sys

# -----------------------------------------------------------
# 调整路径：脚本现在在 docs/ 文件夹内运行
# -----------------------------------------------------------
# 获取当前脚本的绝对目录 (docs/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置文件路径: 脚本目录向上两级 (到项目根目录)
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'config.yaml')
# 目标文档路径: 脚本自身所在目录
ARCH_DOC_FILE = os.path.join(SCRIPT_DIR, 'architecture.md')

# 将项目根目录添加到 sys.path (方便导入其他模块)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))


def load_config():
    """读取 YAML 配置"""
    # 始终使用绝对路径进行文件查找
    if not os.path.exists(CONFIG_FILE_PATH):
        raise FileNotFoundError(f"❌ 找不到配置文件: {CONFIG_FILE_PATH}")
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    """回写 YAML 配置 (更新状态)"""
    with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# ... (update_architecture_status 和后续函数保持不变) ...
def update_architecture_status(config):
    """
    核心功能：将最新的部署状态写入 docs/architecture.md 中特定的标记区域。
    """
    status = config.get('deployment_status', {})
    last_run = status.get('last_run', 'N/A')
    
    # 状态图标映射
    icons = {
        "Pending": "⏳",
        "Success": "✅",
        "Failed": "❌",
        "Running": "🔄"
    }

    # 生成 Markdown 表格内容 (使用 f-string)
    status_content = f"""

_最后一次运行时间: {last_run}_

| 组件 (Component) | 状态 (Status) | 详情 (Details) |
| :--- | :--- | :--- |
| **S3 Storage** | {icons.get(status.get('s3_bucket', 'Pending'))} {status.get('s3_bucket')} | Bucket: `{config['s3']['bucket_name']}` |
| **SageMaker Endpoint** | {icons.get(status.get('model_endpoint', 'Pending'))} {status.get('model_endpoint')} | Name: `{config['sagemaker']['endpoint_name']}` |
| **API Gateway** | {icons.get(status.get('api_gateway', 'Pending'))} {status.get('api_gateway')} | URL: `{status.get('api_url', 'N/A')}` |
| **Frontend App** | {icons.get(status.get('frontend', 'Pending'))} {status.get('frontend')} | Local: `http://localhost:{config['frontend']['port']}` |

"""
    
    # 读取现有的文档
    if not os.path.exists(ARCH_DOC_FILE):
        print(f"⚠️ 警告: 找不到文档 {ARCH_DOC_FILE}。请确保文件存在。", file=sys.stderr)
        return

    with open(ARCH_DOC_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定义标记
    start_marker = "<!-- DEPLOYMENT_STATUS_START -->"
    end_marker = "<!-- DEPLOYMENT_STATUS_END -->"

    # 查找标记位置
    if start_marker in content and end_marker in content:
        start_index = content.find(start_marker) + len(start_marker)
        end_index = content.find(end_marker)
        
        # 构造新内容：前缀 + 标记 + 动态内容 + 标记 + 后缀
        new_content = (
            content[:start_index] + 
            status_content + 
            content[end_index:]
        )
        
        with open(ARCH_DOC_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"📄 [Docs] {ARCH_DOC_FILE} 已更新最新的部署状态。")
    else:
        print(f"❌ 文档中缺少标记 {start_marker} 或 {end_marker}。请按要求修改文档。", file=sys.stderr)

# ... (step_1_check_resources, step_2_deploy_endpoint, step_3_deploy_api 保持不变) ...

def step_1_check_resources(config):
    print("\n--- [Step 1] 检查 AWS 资源 ---")
    time.sleep(0.5) 
    config['deployment_status']['s3_bucket'] = "Success"
    return config

def step_2_deploy_endpoint(config):
    print("\n--- [Step 2] 部署 SageMaker Endpoint ---")
    time.sleep(0.5) 
    config['deployment_status']['model_endpoint'] = "Success"
    return config

def step_3_deploy_api(config):
    print("\n--- [Step 3] 配置 API Gateway & Lambda ---")
    time.sleep(0.5)
    fake_url = "https://xyz123.execute-api.us-east-1.amazonaws.com/prod/predict"
    config['deployment_status']['api_gateway'] = "Success"
    config['deployment_status']['api_url'] = fake_url
    return config


def main():
    # 提醒用户从根目录运行
    print("🤖 初始化部署管家 (Deployment Manager)...")
    config = load_config()
    
    # 更新开始时间
    config['deployment_status']['last_run'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 执行步骤链
        config = step_1_check_resources(config)
        update_architecture_status(config) # 实时更新文档
        
        config = step_2_deploy_endpoint(config)
        update_architecture_status(config)
        
        config = step_3_deploy_api(config)
        update_architecture_status(config)
        
        print("\n✨ 所有部署步骤完成！请查看 docs/architecture.md 获取最新状态。")
        
    except Exception as e:
        print(f"\n❌ 部署中断: {e}")
    finally:
        save_config(config)

if __name__ == "__main__":
    main()