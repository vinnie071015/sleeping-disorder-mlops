(mlops-env) zhengchengsheng@MacBook-Air-2 sleeping_disorder_prediction % tree -L 3
.
├── README.md
├── adjustment_for_project.ipynb
├── api
│   ├── app.py
│   └── request_schema.json
├── config.yaml
├── data
│   └── Sleep_health_and_lifestyle_dataset.csv
├── docker
│   └── Dockerfile
├── docs
│   ├── architecture.md
│   ├── deploy_manager.py
│   ├── model_metadata.json
│   └── temp_operation_record.md
├── notebooks
│   ├── 01_sagemaker_orchestration.ipynb
│   ├── best_model_extracted
│   │   ├── label_encoder.joblib
│   │   └── model.joblib
│   ├── debug_logger_test.py
│   └── model.tar.gz
├── requirements.txt
├── s3_log_uploader.py
├── scripts
│   ├── debug_sagemaker_perms.py
│   └── upload_raw_data.py
├── setup_skeleton.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── data_processor.cpython-39.pyc
│   │   └── train.cpython-39.pyc
│   ├── data_processor.py
│   ├── requirements.txt
│   └── train.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-39.pyc
    │   ├── test_data.cpython-39-pytest-8.4.2.pyc
    │   └── test_training.cpython-39-pytest-8.4.2.pyc
    ├── test_data.py
    └── test_training.py

12 directories, 34 files

# 替换为您的 AWS 账号 ID
export AWS_ACCOUNT_ID=137568342316
export AWS_REGION=us-east-1
export ECR_REPO_NAME=sleep-predictor

# 1. 登录 ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 2. 标记本地镜像
docker tag sleep-predictor:v1 ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:v1

# 3. 推送镜像到 ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:v1

aws iam attach-user-policy --user-name ML_Project --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

{
    "UserId": "AIDASAB5WKEWLFSFMA3PR",
    "Account": "137568342316",
    "Arn": "arn:aws:iam::137568342316:user/ML_Project"
}

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=137568342316
export SAGEMAKER_ROLE=arn:aws:iam::137568342316:role/SageMakerExecutionRole
export BUCKET_NAME=sleep-disorder-mlops-bucket
export MODEL_ARTIFACT_KEY=sagemaker-tuning-output/svm-tuning-251203-2103-002-c8ccdacc/output/model.tar.gz
export MODEL_NAME=sleep-disorder-svm-model-v1 
export ECR_IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/sleep-predictor:v1

步骤 2：创建 SageMaker Model
使用您推送的 ECR 镜像和 S3 模型路径创建新的 SageMaker Model 资源：

Bash

aws sagemaker create-model \
    --model-name ${MODEL_NAME} \
    --execution-role-arn ${SAGEMAKER_ROLE} \
    --primary-container Image=${ECR_IMAGE},ModelDataUrl=s3://${BUCKET_NAME}/${MODEL_ARTIFACT_KEY} \
    --region ${AWS_REGION}
--model-name: 我们创建了一个新的模型名称 (sleep-disorder-svm-model-v1)。

Image: 指向您刚刚推送到 ECR 的镜像。

ModelDataUrl: 指向您 S3 上最优模型的 model.tar.gz 路径。

## -----
# 确保所有环境变量已定义 (参考上一个回复)

# 步骤 2：创建 SageMaker Model
aws sagemaker create-model \
    --model-name ${MODEL_NAME} \
    --execution-role-arn ${SAGEMAKER_ROLE} \
    --primary-container Image=${ECR_IMAGE},ModelDataUrl=s3://${BUCKET_NAME}/${MODEL_ARTIFACT_KEY} \
    --region ${AWS_REGION}

echo ${SAGEMAKER_ROLE}：
arn:aws:iam::137568342316:role/SageMakerExecutionRole
aws iam get-role --role-name SageMakerExecutionRole：
arn:aws:iam::137568342316:role/SageMakerExecutionRole


# 步骤 3：创建 Endpoint Configuration
export ENDPOINT_CONFIG_NAME=sleep-disorder-svm-config-v1

aws sagemaker create-endpoint-config \
    --endpoint-config-name ${ENDPOINT_CONFIG_NAME} \
    --production-variants VariantName=AllTraffic,ModelName=${MODEL_NAME},InitialInstanceCount=1,InstanceType=ml.t2.medium \
    --region ${AWS_REGION}

# 步骤 4：更新 Endpoint (开始部署)
export ENDPOINT_NAME=sleep-disorder-svm-prod-v1

aws sagemaker update-endpoint \
    --endpoint-name ${ENDPOINT_NAME} \
    --endpoint-config-name ${ENDPOINT_CONFIG_NAME} \
    --region ${AWS_REGION}


aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly


    aws ecr set-repository-policy \
    --repository-name ${ECR_REPO_NAME} \
    --policy-text '{"Version": "2008-10-17", "Statement": [ { "Sid": "AllowSameAccountPull", "Effect": "Allow", "Principal": { "AWS": "arn:aws:iam::137568342316:root" }, "Action": [ "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage" ] } ] }' \
    --region ${AWS_REGION}

