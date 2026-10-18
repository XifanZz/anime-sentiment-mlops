import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. 读取数据
data_path = "data/anime_sentiment_balanceado.csv"
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# 检查列名，假设是 texto_resena 和 sentimiento
# 如果你的列名不一样，请在这里修改
X = df['texto_resena']
y = df['sentimiento']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 创建 Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(random_state=42))
])

# 3. MLflow 实验记录
mlflow.set_experiment("Anime_Sentiment_Baseline")

with mlflow.start_run(run_name="TFIDF_LogReg_v1"):
    print("Training Baseline model...")
    
    # 记录 Git Commit Hash (如果还没初始化 git，这一步会返回 None)
    commit_hash = os.popen('git rev-parse HEAD').read().strip()
    if commit_hash:
        mlflow.log_param("git_commit", commit_hash)
        
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 记录参数和指标
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # 记录模型
    mlflow.sklearn.log_model(pipeline, "baseline_model")
    
    print("✅ Training complete!")
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")