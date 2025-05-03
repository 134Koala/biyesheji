import jqdatasdk as jq
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 聚宽认证（替换为你的账号密码）
jq.auth("13145227025", "Zhujunhe.113807")

# 获取平安银行(000001.XSHE)历史数据 [6,7](@ref)
df = jq.get_price(
    "000001.XSHE",
    start_date="2024-01-01",
    end_date="2025-01-26",
    frequency="daily",
    fields=['close']  # 以收盘价为基准
)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])

# 创建时间窗口数据集（60天预测1天）[1,5](@ref)
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 转换为3D输入 (样本数, 时间步, 特征数)

# 划分训练集和测试集
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]