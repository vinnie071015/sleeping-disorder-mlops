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

@app.on_event("startup")
def load_artifacts():
    global model_pipeline, label_encoder
    # 假设在 Docker 中模型位于 /opt/ml/model 目录
    # 在本地测试时，请确保这两个文件在当前目录下或修改路径
    model_path = os.getenv("MODEL_PATH", "model.joblib")
    le_path = os.getenv("LE_PATH", "label_encoder.joblib")
    
    try:
        model_pipeline = joblib.load(model_pipeline_path)
        label_encoder = joblib.load(label_encoder_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 本地调试时的备用路径 (根据您的 notebook 结构)
        try:
            base_path = "../notebooks/best_model_extracted/"
            model_pipeline = joblib.load(os.path.join(base_path, "model.joblib"))
            label_encoder = joblib.load(os.path.join(base_path, "label_encoder.joblib"))
            print("✅ 模型加载成功 (本地调试路径)")
        except Exception as e2:
            print(f"❌ 本地路径也加载失败: {e2}")

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