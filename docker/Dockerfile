# 1. 使用官方 Python 3.9 轻量级镜像作为基础，并强制指定架构
# 修正：添加 --platform=linux/amd64 解决 MacBook M1/M2 的架构兼容性问题
FROM --platform=linux/amd64 python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 设置环境变量，防止 Python 缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 复制依赖清单并安装
# 修正：使用根目录下的 requirements.txt
COPY requirements.txt .
# 额外的库保持安装，但建议将它们也放入 requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn joblib boto3 pandas numpy

# 5. 复制核心代码和模型文件 (Build Context 为项目根目录)
COPY api/ api/
COPY src/ src/

# ⚠️ FIX: 复制模型和编码器文件到容器内的 /app 目录
# 目标路径： /app/notebooks/best_model_extracted/
COPY notebooks/best_model_extracted/ notebooks/best_model_extracted/

# 6. 设置 SageMaker 标准环境变量
# 修正：将模型路径设置为 SageMaker 运行时挂载的默认路径
ENV MODEL_DIR=/opt/ml/model
ENV CODE_DIR=/app

# 7. 暴露端口 (SageMaker 默认使用 8080)
EXPOSE 8080

# 8. 启动命令: 使用 uvicorn 启动 FastAPI 服务
# 路径修正：确保 api.app:app 正确指向你的启动文件
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]