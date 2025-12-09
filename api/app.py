import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

# 1. Define Request Data Format (based on request_schema.json)
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

# 2. Initialize FastAPI Application
app = FastAPI(title="Sleep Disorder Prediction API")
# Global variables for loading the model
model_pipeline = None
label_encoder = None

# [IMPORTANT: Unify Model Path Variable Name]
# SageMaker mounts the model at MODEL_DIR (i.e., /opt/ml/model)
MODEL_DIR = os.getenv("MODEL_DIR", ".") 

@app.on_event("startup")
def load_artifacts():
    # ⚠️ Fix 1: global declaration must be at the beginning of the function
    global model_pipeline, label_encoder 
    
    # Correction: Use SageMaker standard path directly (model files are after tarball decompression)
    MODEL_FILENAME = "model.joblib"
    LE_FILENAME = "label_encoder.joblib"
    
    # Correction: Use the unified MODEL_DIR environment variable
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    le_path = os.path.join(MODEL_DIR, LE_FILENAME)
    
    # --- Attempt 1: SageMaker/EC2 Standard Path Loading ---
    try:
        model_pipeline = joblib.load(model_path)
        label_encoder = joblib.load(le_path)
        print("✅ Model and encoder loaded successfully (Path 1: MODEL_DIR)")
        return
    except Exception as e:
        print(f"❌ Model loading failed (Path 1: MODEL_DIR): {e}")

    # --- Attempt 2: Container Internal Absolute Path Loading (for Docker run debugging) ---
    try:
        # Absolute path: /app/notebooks/best_model_extracted/
        base_path = "/app/notebooks/best_model_extracted" 
        
        model_pipeline = joblib.load(os.path.join(base_path, "model.joblib"))
        label_encoder = joblib.load(os.path.join(base_path, "label_encoder.joblib"))
        print("✅ Model loaded successfully (Path 2: Container Absolute Path)")
        return
    except Exception as e2:
        print(f"❌ Model loading failed (Path 2: Container Absolute Path): {e2}")
        
    # --------------------------------------------------------------------------
    # ⚠️ Fix 2: Clear syntax error, set model_pipeline = None if model fails to load
    # --------------------------------------------------------------------------
    # Ensure model_pipeline can be referenced even if loading fails (but the value is None)
    # model_pipeline and label_encoder were declared as global at the start of the function
    model_pipeline = None 
    label_encoder = None

    print("⚠️ Warning: Model failed to load, but the server will start and respond to /ping request (returning 500).")
    return

@app.get("/ping")
def health_check():
    """Health check interface required by AWS SageMaker"""
    # Fix: Ensure model_pipeline is correctly referenced (no need for global keyword)
    if model_pipeline is not None:
        return {"status": "Healthy"}
    raise HTTPException(status_code=500, detail="Model not loaded")

@app.post("/invocations")
async def predict(input_data: SleepInput):
    """Inference interface required by AWS SageMaker (path must be /invocations)"""
    # Ensure model_pipeline and label_encoder are correctly referenced
    if not model_pipeline or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # 1. Convert to DataFrame (This is the input format expected by the Pipeline)
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # 2. Perform prediction
        pred_encoded = model_pipeline.predict(df)
        
        # 3. Decode result (0 -> Insomnia)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]
        
        return {"prediction": pred_label}
        
    except Exception as e:
        # Print detailed Python error information for easy debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# Local testing startup command: uvicorn api.app:app --reload