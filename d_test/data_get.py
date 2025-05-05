import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(symbol, start_date, end_date, look_back=60, forecast_steps=3):
    """
    获取并预处理股票数据（使用akshare数据源）
    :return: (X_train, y_train, X_test, y_test, scaler, forecast_steps)
    """
    # 转换日期格式为YYYYMMDD
    start_date = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_date = pd.to_datetime(end_date).strftime("%Y%m%d")
    
    # 获取A股历史行情数据
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 使用前复权数据
    )
    print("实际获取的列名:", df.columns.tolist())  # 检查语句
    # 重命名列并设置索引
    df.rename(columns={
        '日期': 'date',
        '收盘': 'close',
        '成交量': 'volume'
    }, inplace=True)
    
    # 转换为日期索引并按日期排序
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)  # 确保时间升序排列
    
    # 划分训练测试集（保持原比例）
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]['close'].values.reshape(-1, 1)
    test_data = df.iloc[train_size:]['close'].values.reshape(-1, 1)

    # 归一化处理
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    # 生成监督数据
    def create_dataset(data, n_steps):
        X, y = [], []
        for i in range(len(data)-n_steps-forecast_steps):
            X.append(data[i:i+n_steps, 0])
            y.append(data[i+n_steps:i+n_steps+forecast_steps, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)

    # 调整维度适应RNN输入
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test = X_test.reshape(*X_test.shape, 1)

    return X_train, y_train, X_test, y_test, scaler, forecast_steps
# if __name__ == "__main__":
#     X_train, y_train, X_test, y_test, scaler, steps = get_stock_data(
#         symbol="000001",  
#         start_date="2020-01-01",
#         end_date="2024-12-31"
#     )
#     print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")