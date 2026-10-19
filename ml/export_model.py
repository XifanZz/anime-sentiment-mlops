import mlflow
import mlflow.sklearn
import os
import joblib

# 设置为本地 mlruns 目录
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Anime_Sentiment_Baseline")

# 找到最新的成功运行的模型
experiment = mlflow.get_experiment_by_name("Anime_Sentiment_Baseline")
if experiment is None:
    raise ValueError("找不到实验，请确认同学A是否已经把 mlruns 文件夹提交到仓库。")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
run_id = runs.iloc[0].run_id

print(f"Loading model from run: {run_id}")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/baseline_model")

# 保存到本地普通 pkl 文件
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_model.pkl")
print("✅ Model exported to models/baseline_model.pkl successfully!")