# 1. 使用官方 Python 3.9 轻量级镜像作为基础，并强制指定架构
# 修正：添加 --platform=linux/amd64 解决 MacBook M1/M2 的架构兼容性问题
FROM --platform=linux/amd64 python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 设置环境变量，防止 Python 缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 复制依赖清单并安装
COPY requirements.txt .
# -----------------------------------------------------------------------
# ⚠️ FIX 1: 移除 COPY api/ api/ 的错误连接。仅保留 RUN pip install。
RUN pip install --no-cache-dir -r requirements.txt
# -----------------------------------------------------------------------

# 5. 复制核心代码 (Build Context 为项目根目录)
# ⚠️ FIX 2: 确保所有的 COPY 都是独立的指令
COPY api/ api/
COPY src/ src/

# 6. 复制模型文件
COPY notebooks/best_model_extracted/ notebooks/best_model_extracted/

# 7. 暴露端口 (SageMaker 默认使用 8080)
EXPOSE 8080

# 8. 启动命令: 使用 uvicorn 启动 FastAPI 服务
# 路径修正：确保 api.app:app 正确指向你的启动文件
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]