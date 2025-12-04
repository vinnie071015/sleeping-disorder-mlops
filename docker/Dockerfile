# 1. 使用官方 Python 3.9 轻量级镜像作为基础
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖清单并安装
# 注意：我们额外安装了 fastapi, uvicorn 和 pydantic，这是运行 API 必需的
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn pydantic

# 4. 复制核心代码
COPY api/ api/
COPY src/ src/

# 5. 复制模型文件到容器根目录
# (假设您已经在 notebooks/best_model_extracted/ 下解压了模型)
COPY notebooks/best_model_extracted/model.joblib .
COPY notebooks/best_model_extracted/label_encoder.joblib .

# 6. 设置环境变量，告诉 app.py 模型在哪里
ENV MODEL_PATH=./model.joblib
ENV LE_PATH=./label_encoder.joblib

# 7. 暴露端口 (SageMaker 默认使用 8080)
EXPOSE 8080

# 8. 启动命令: 使用 uvicorn 启动 FastAPI 服务
# --host 0.0.0.0 允许外部访问
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]