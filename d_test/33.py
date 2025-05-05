import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rolling_forecast(model, initial_data, steps):
    predictions = []
    current_batch = initial_data.reshape(1, *initial_data.shape, 1)
    for _ in range(steps):
        pred = model.predict(current_batch)[0][0]
        predictions.append(pred)
        current_batch = np.concatenate(
            [current_batch[:,1:,:], [[[pred]]]], axis=1
        )
    return np.array(predictions)

def main():
    # 加载模型和数据
    from model import build_model
    from d_test.data_processor import get_stock_data
    _, _, X_test, y_test, scaler, forecast_steps = get_stock_data(
        "000001", "2020-01-01", "2024-12-31"
    )
    
    model = build_model((X_test.shape[1], 1), forecast_steps)
    model.load_weights("best_model.h5")
    
    # 预测
    y_pred = model.predict(X_test)
    last_window = X_test[-1,:,0]
    y_rolled = rolling_forecast(model, last_window, len(y_test))
    
    # 反归一化
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
    y_rolled = scaler.inverse_transform(y_rolled.reshape(-1,1))
    
    # 可视化
    plt.figure(figsize=(14,6))
    plt.plot(y_true, label='Actual', alpha=0.6)
    plt.plot(y_pred, '--', label='Static Forecast')
    plt.plot(y_rolled, '-.', label='Rolling Forecast')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 评估指标
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()