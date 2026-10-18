FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件并安装（利用 Docker 缓存机制）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动命令（假设 app 文件夹下有个 main.py，里面有 app 实例）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]