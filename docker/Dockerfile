# 1. 基础镜像
FROM --platform=linux/amd64 python:3.9-slim

# 2. 工作目录
WORKDIR /app

# 3. 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 依赖安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制核心代码
COPY api/ api/
COPY src/ src/
# --- 新增：复制前端代码 ---
COPY frontend/ frontend/ 

# 6. 复制模型文件
COPY notebooks/best_model_extracted/ notebooks/best_model_extracted/

# 7. --- 新增：复制并设置启动脚本 ---
COPY run_services.sh .
RUN chmod +x run_services.sh

# 8. --- 修改：暴露两个端口 ---
# 8000 是 API, 8501 是 Streamlit 页面
EXPOSE 8000 8501

# 9. --- 修改：启动命令改为运行脚本 ---
CMD ["./run_services.sh"]