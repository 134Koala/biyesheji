import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataProcessor:
    def __init__(self, symbol, start_date, end_date, test_size=0.2, look_back=60):
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.scaler = MinMaxScaler()
        self.look_back = look_back
        
    def prepare_data(self):
        # 获取原始数据
        df = ak.stock_zh_a_hist(
            symbol=self.symbol,
            period="daily",
            start_date=self.start_date.strftime("%Y%m%d"),
            end_date=self.end_date.strftime("%Y%m%d"),
            adjust="qfq"
        )
        
        # 数据预处理
        df = df[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # # 数据集划分
        # split_idx = int(len(df) * (1 - 0.2))
        # self.train_data = df.iloc[:split_idx]
        # self.test_data = df.iloc[split_idx - self.look_back:]  # 保留look_back用于窗口
        
        # 在DataProcessor中划分训练/验证/测试集
        split_idx = int(len(df) * 0.7)
        val_idx = int(len(df) * 0.85)
        self.train_data = df.iloc[:split_idx]
        self.val_data = df.iloc[split_idx:val_idx]
        self.test_data = df.iloc[val_idx - self.look_back:]

        # 归一化处理
        self.scaler.fit(self.train_data[['close']])
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return self.train_data, self.val_data, self.test_data  # 新增验证集返回
    
    def create_dataset(self, data):
        scaled = self.scaler.transform(data[['close']])
        X = []  # 添加初始化
        y = []  # 添加初始化
        for i in range(len(scaled) - self.look_back - 1):
            X.append(scaled[i:i+self.look_back, 0])
            y.append(scaled[i+self.look_back, 0])
        return np.array(X), np.array(y)