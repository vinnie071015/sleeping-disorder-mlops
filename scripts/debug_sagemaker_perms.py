import boto3
import json
import os
import sys

# 颜色代码，方便查看
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_pass(msg):
    print(f"{GREEN}[PASS] {msg}{RESET}")

def print_fail(msg):
    print(f"{RED}[FAIL] {msg}{RESET}")

def print_warn(msg):
    print(f"{YELLOW}[WARN] {msg}{RESET}")

def check_environment():
    print(f"\n--- 1. 检查环境变量 ---")
    required_vars = ["SAGEMAKER_ROLE", "ECR_IMAGE", "AWS_REGION", "BUCKET_NAME", "MODEL_ARTIFACT_KEY"]
    env_data = {}
    missing = []
    
    for var in required_vars:
        val = os.environ.get(var)
        if not val:
            missing.append(var)
        else:
            env_data[var] = val
            print(f"{var}: {val}")
    
    if missing:
        print_fail(f"缺少环境变量: {missing}")
        sys.exit(1)
    return env_data

def check_iam_role(iam_client, role_arn):
    print(f"\n--- 2. 验证 IAM Role (身份与信任) ---")
    role_name = role_arn.split("/")[-1]
    
    try:
        response = iam_client.get_role(RoleName=role_name)
        role = response['Role']
        print_pass(f"Role '{role_name}' 存在。")
        
        # 检查信任关系
        assume_policy = role['AssumeRolePolicyDocument']
        trusts_sagemaker = False
        print(f"Trust Policy:\n{json.dumps(assume_policy, indent=2)}")
        
        for statement in assume_policy['Statement']:
            principal = statement.get('Principal', {})
            service = principal.get('Service', [])
            if isinstance(service, str):
                service = [service]
            
            if "sagemaker.amazonaws.com" in service:
                trusts_sagemaker = True
                break
        
        if trusts_sagemaker:
            print_pass("Trust Relationship 正确: 包含 sagemaker.amazonaws.com")
        else:
            print_fail("Trust Relationship 错误: 未发现 sagemaker.amazonaws.com！SageMaker 无法扮演此角色。")
            
    except Exception as e:
        print_fail(f"获取 Role 信息失败: {str(e)}")

def check_ecr_image(ecr_client, image_uri):
    print(f"\n--- 3. 验证 ECR 镜像 (资源与架构) ---")
    # 解析 URI: 12345.dkr.ecr.region.amazonaws.com/repo-name:tag
    try:
        repo_part = image_uri.split("/", 1)[1]
        repo_name, image_tag = repo_part.split(":")
    except ValueError:
        print_fail(f"ECR URI 格式解析失败: {image_uri}")
        return

    print(f"Repository: {repo_name}")
    print(f"Tag: {image_tag}")

    try:
        # 1. 检查镜像是否存在
        response = ecr_client.batch_get_image(
            repositoryName=repo_name,
            imageIds=[{'imageTag': image_tag}]
        )
        
        if not response['images']:
            print_fail(f"在仓库 {repo_name} 中找不到标签为 {image_tag} 的镜像！")
            if response['failures']:
                print_fail(f"Failure Reason: {response['failures'][0].get('failureCode')}")
            return
        
        print_pass("镜像存在于 ECR 中。")
        
        # 2. 检查架构 (Architecture) - 关键点！
        # 注意：DescribeImages 能看到更详细的 manifest
        desc_resp = ecr_client.describe_images(
            repositoryName=repo_name,
            imageIds=[{'imageTag': image_tag}]
        )
        # 这里只是简单的检查，真正确定的架构需要在 manifest 里面看，
        # 但如果之前是 Mac 构建且未加 --platform，这里往往能看出端倪
        print_warn("请注意：必须确保镜像是 linux/amd64。")
        
    except Exception as e:
        print_fail(f"检查 ECR 镜像失败: {str(e)}")

    # 3. 检查 Repository Policy
    try:
        policy_resp = ecr_client.get_repository_policy(repositoryName=repo_name)
        policy_text = json.loads(policy_resp['policyText'])
        print(f"Repo Policy:\n{json.dumps(policy_text, indent=2)}")
        print_pass("成功获取 ECR Repository Policy (请人工检查是否允许了 Role 或 Root)。")
    except ecr_client.exceptions.RepositoryPolicyNotFoundException:
        print_warn("未设置 Repository Policy (默认为私有，仅允许本账号 IAM 访问)。如果 Role 和 ECR 在同账号，这通常没问题。")
    except Exception as e:
        print_fail(f"获取 Repository Policy 失败: {str(e)}")

def check_s3_artifact(s3_client, bucket, key):
    print(f"\n--- 4. 验证 S3 模型文件 ---")
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        print_pass(f"模型文件 s3://{bucket}/{key} 存在。")
    except Exception as e:
        print_fail(f"无法访问 S3 对象: {str(e)}。SageMaker 将无法下载模型。")

def main():
    try:
        env = check_environment()
        
        session = boto3.Session(region_name=env['AWS_REGION'])
        iam = session.client('iam')
        ecr = session.client('ecr')
        s3 = session.client('s3')
        
        check_iam_role(iam, env['SAGEMAKER_ROLE'])
        check_ecr_image(ecr, env['ECR_IMAGE'])
        check_s3_artifact(s3, env['BUCKET_NAME'], env['MODEL_ARTIFACT_KEY'])
        
        print(f"\n--- 诊断结束 ---")
        print("如果以上所有检查都为 [PASS]，但依然报错，可能涉及：")
        print("1. KMS 加密权限缺失 (如果 ECR 或 S3 使用了自定义 KMS Key)。")
        print("2. VPC Endpoint 限制 (如果使用了 Private VPC)。")
        
    except Exception as e:
        print_fail(f"脚本执行出错: {str(e)}")

if __name__ == "__main__":
    main()