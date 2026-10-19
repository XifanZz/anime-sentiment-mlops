from fastapi.testclient import TestClient
from app.main import app, model
import app.main as main_module

# 💡 DevOps 技巧：在 CI 环境中，模型文件通常不存在。
# 我们在这里注入一个 Mock 模型，专门用来测试 API 逻辑，而不依赖真实的 .pkl 文件。
class DummyModel:
    def predict(self, texts):
        return [1]  # 永远预测为正面
    def predict_proba(self, texts):
        return [[0.1, 0.9]]  # 永远返回 90% 置信度

# 将主程序里的 model 替换为假模型
main_module.model = DummyModel()

client = TestClient(app)

def test_health_check():
    """测试 1：健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # 因为是 DummyModel，所以这里肯定不为 None
    assert response.json()["model_loaded"] is True

def test_predict_positive():
    """测试 2：预测正面情感"""
    response = client.post("/predict", json={"text": "This anime is a masterpiece!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5

def test_predict_negative():
    """测试 3：预测（假模型也会返回正面，我们主要测接口通不通）"""
    response = client.post("/predict", json={"text": "Worst anime ever."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"  # DummyModel 返回的是 1
    assert data["confidence"] > 0.5

def test_predict_empty_text():
    """测试 4（边界情况）：空文本"""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "Text cannot be empty" in response.json()["detail"]