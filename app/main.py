from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Anime Sentiment Analyzer API")

# 加载模型
MODEL_PATH = "models/baseline_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: Model not found!")

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

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