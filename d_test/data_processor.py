# 文件1：data_processor.py（原data_get.py重构）
import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(symbol, start_date, end_date, look_back=60, forecast_steps=1):
    """数据获取和预处理专用模块"""
    # 日期格式转换
    start_date = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_date = pd.to_datetime(end_date).strftime("%Y%m%d")
    
    # 获取数据
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    
    # 列重命名和索引设置
    df.rename(columns={'日期': 'date', '收盘': 'close'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # 只返回完整数据集和scaler
    return df[['close']], MinMaxScaler()

def create_dataset(data, scaler, look_back, forecast_steps):
    """数据集生成函数"""
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data)-look_back-forecast_steps):
        X.append(scaled_data[i:i+look_back, 0])
        y.append(scaled_data[i+look_back:i+look_back+forecast_steps, 0])
        
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y, scaler