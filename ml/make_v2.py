import pandas as pd
import os

def main():
    v1_path = "data/anime_sentiment_balanceado.csv"
    v2_path = "data/anime_sentiment_v2.csv"
    
    print("Loading v1 dataset...")
    if not os.path.exists(v1_path):
        raise FileNotFoundError(f"Cannot find {v1_path}. Did you copy it from A?")
        
    df = pd.read_csv(v1_path)
    print(f"Original v1 shape: {df.shape}")
    
    # 改进策略 (模拟清理)：
    # 1. 过滤掉长度小于 50 个字符的极短评论
    # 2. 随机抽样 5000 条，方便后面快速跑 BERT 微调
    print("Filtering and sampling to create v2...")
    df_cleaned = df[df['texto_resena'].astype(str).str.len() > 50]
    df_v2 = df_cleaned.sample(n=5000, random_state=42)
    
    df_v2.to_csv(v2_path, index=False)
    print(f"Created v2 dataset at {v2_path} with shape: {df_v2.shape}")

if __name__ == "__main__":
    main()
