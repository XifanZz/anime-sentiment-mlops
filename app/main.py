import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 1. 初始化 FastAPI 应用
app = FastAPI(title="Anime Sentiment Analyzer API")

# 2. 加载模型
MODEL_PATH = "models/baseline_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: Model not found!")

# 3. 定义数据模型 (Schemas)
class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

# 4. 挂载静态文件目录 (前端页面用)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 5. 定义 API 路由
@app.get("/")
def serve_frontend():
    # 返回前端 HTML 页面
    return FileResponse("app/static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 边界情况：空文本
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # 预测
    prediction = model.predict([request.text])[0]
    probability = model.predict_proba([request.text])[0]
    
    # 获取置信度
    confidence = max(probability)
    sentiment = "positive" if prediction == 1 else "negative"
    
    return PredictionResponse(sentiment=sentiment, confidence=round(float(confidence), 4))