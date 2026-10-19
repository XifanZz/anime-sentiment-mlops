from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_check():
    """测试 1：健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_positive():
    """测试 2：预测正面情感"""
    response = client.post("/predict", json={"text": "This anime is a masterpiece! I love it so much."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5

def test_predict_negative():
    """测试 3：预测负面情感"""
    response = client.post("/predict", json={"text": "Worst anime ever. Terrible plot and boring characters."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.5

def test_predict_empty_text():
    """测试 4（边界情况）：空文本"""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "Text cannot be empty" in response.json()["detail"]