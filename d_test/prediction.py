import matplotlib.pyplot as plt
from model import *
# 加载最佳模型
model.load_weights("best_model.h5")

# 测试集预测
y_pred = model.predict(dg.X_test)
y_pred = dg.scaler.inverse_transform(y_pred)  # 反归一化
y_true = dg.scaler.inverse_transform(dg.y_test.reshape(-1,1))

# 结果可视化 [4,5](@ref)
plt.figure(figsize=(14,6))
plt.plot(y_true, label='Actual Price', color='blue', alpha=0.6)
plt.plot(y_pred, label='Predicted Price', color='red', linestyle='--')
plt.title('PingAn Bank Stock Price Prediction')
plt.xlabel('Trading Days')
plt.ylabel('Price (CNY)')
plt.legend()
plt.grid(True)
plt.show()

# 评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")