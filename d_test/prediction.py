# 文件3：predict.py（重构后的预测模块）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rolling_forecast(model, initial_data, steps):
    """滚动预测函数保持不变"""
    predictions = []
    current_batch = initial_data.reshape(1, *initial_data.shape, 1)
    for _ in range(steps):
        pred = model.predict(current_batch)[0]
        predictions.append(pred[0])  # 仅取最后一步预测
        current_batch = np.concatenate(
            [current_batch[:,1:,:], pred.reshape(1,1,1)], axis=1
        )
    return np.array(predictions)

def main():
    """主预测流程"""
    # 加载训练好的组件
    from model import build_model, train_model,prepare_datasets
    
    # 使用训练时生成的测试集
    model, X_test, y_test, scaler = train_model()  # 实际应持久化保存这些数据
    
    # 预测
    y_pred = model.predict(X_test)
    last_window = X_test[-1,:,0]
    y_rolled = rolling_forecast(model, last_window, len(y_test))
    
    # 反归一化
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
    y_rolled = scaler.inverse_transform(y_rolled.reshape(-1,1))
    
    # 可视化与评估
    plt.figure(figsize=(14,6))
    plt.plot(y_true, label='Actual', alpha=0.6)
    plt.plot(y_pred, '--', label='Static Forecast')
    plt.plot(y_rolled, '-.', label='Rolling Forecast')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()