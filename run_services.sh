#!/bin/bash

# 1. 在后台启动 FastAPI (端口 8000)
# nohup 表示不挂断运行，& 表示后台运行
echo "Starting FastAPI backend..."
nohup uvicorn api.app:app --host 0.0.0.0 --port 8000 > /app/backend.log 2>&1 &

# 2. 等待几秒钟，确保后端启动
sleep 5

# 3. 在前台启动 Streamlit (端口 8501)
echo "Starting Streamlit frontend..."
streamlit run frontend/ui.py --server.port 8501 --server.address 0.0.0.0