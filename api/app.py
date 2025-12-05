import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. 定义请求数据格式 (基于 request_schema.json)
class SleepInput(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    quality_of_sleep: int
    physical_activity_level: int
    stress_level: int
    bmi_category: str
    blood_pressure: str
    heart_rate: int
    daily_steps: int

# 2. 初始化 FastAPI 应用
app = FastAPI(title="Sleep Disorder Prediction API")
# 全局变量加载模型
model_pipeline = None
label_encoder = None

# 【重要：统一模型路径变量名】
# SageMaker 将模型挂载在 MODEL_DIR (即 /opt/ml/model)
MODEL_DIR = os.getenv("MODEL_DIR", ".") 

@app.on_event("startup")
def load_artifacts():
    global model_pipeline, label_encoder
    
    # 修正：直接使用 SageMaker 标准路径 (模型文件在 tarball 解压后)
    MODEL_FILENAME = "model.joblib"
    LE_FILENAME = "label_encoder.joblib"
    
    # 修正：使用统一的 MODEL_DIR 环境变量
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    le_path = os.path.join(MODEL_DIR, LE_FILENAME)
    
    try:
        # 尝试从 SageMaker 标准路径加载 (本地测试时会失败，但 CI/CD 部署到 SageMaker/EC2 时可能有用)
        model_pipeline = joblib.load(model_path)
        label_encoder = joblib.load(le_path)
        print("✅ 模型和编码器加载成功 (路径 1: MODEL_DIR)")
        return
    except Exception as e:
        print(f"❌ 模型加载失败 (路径 1: MODEL_DIR): {e}")

    # --------------------------------------------------------------------------
    # ⚠️ FIX: 尝试从容器内部的绝对路径加载 (当使用 Docker run 时，模型在此处)
    # --------------------------------------------------------------------------
    try:
        # 绝对路径: 从容器的根目录 /app 开始，匹配 Dockerfile 的 COPY 目标
        # 路径为： /app/notebooks/best_model_extracted/
        base_path = "/app/notebooks/best_model_extracted" 
        
        model_pipeline = joblib.load(os.path.join(base_path, "model.joblib"))
        label_encoder = joblib.load(os.path.join(base_path, "label_encoder.joblib"))
        print("✅ 模型加载成功 (路径 2: 容器绝对路径)")
        return
    except Exception as e2:
        print(f"❌ 模型加载失败 (路径 2: 容器绝对路径): {e2}")
        
    # 如果两个路径都失败，抛出致命错误
    raise RuntimeError(f"致命错误：无法加载模型，请检查 Dockerfile COPY 路径是否正确。错误详情: {e2}")

@app.get("/ping")
def health_check():
    """AWS SageMaker 需要的健康检查接口"""
    if model_pipeline is not None:
        return {"status": "Healthy"}
    raise HTTPException(status_code=500, detail="Model not loaded")

@app.post("/invocations")
async def predict(input_data: SleepInput):
    """AWS SageMaker 需要的推理接口 (路径必须是 /invocations)"""
    if not model_pipeline or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # 1. 转换为 DataFrame (这是 Pipeline 预期的输入格式)
        # 注意：这里需要手动处理列名，使其与训练时的标准化列名一致
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # 2. 进行预测
        pred_encoded = model_pipeline.predict(df)
        
        # 3. 解码结果 (0 -> Insomnia)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]
        
        return {"prediction": pred_label}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 本地测试启动命令: uvicorn api.app:app --reload