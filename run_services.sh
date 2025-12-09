#!/bin/bash

# 1. Start FastAPI in the background (Port 8000)
# nohup means running without hangup, & means running in the background
echo "Starting FastAPI backend..."
nohup uvicorn api.app:app --host 0.0.0.0 --port 8000 > /app/backend.log 2>&1 &

# 2. Wait a few seconds to ensure the backend has started
sleep 5

# 3. Start Streamlit in the foreground (Port 8501)
echo "Starting Streamlit frontend..."
streamlit run frontend/ui.py --server.port 8501 --server.address 0.0.0.0