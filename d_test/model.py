# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Dropout
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import GRU, Dense, Dropout
from keras.api._v2.keras.callbacks import ModelCheckpoint, EarlyStopping
import data_get as dg

model = Sequential()
# 第一层GRU（返回完整序列供下一层处理）[1,5](@ref)
model.add(GRU(128, return_sequences=True, input_shape=(dg.X_train.shape[1], 1)))
model.add(Dropout(0.3))  # 随机失活防止过拟合
# 第二层GRU（仅返回最后输出）[2,4](@ref)
model.add(GRU(64))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']  # 监控平均绝对误差
)

# 回调函数配置 [5](@ref)
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# 模型训练
history = model.fit(
    dg.X_train, dg.y_train,
    epochs=100,
    batch_size=32,
    validation_data=(dg.X_test, dg.y_test),
    callbacks=[checkpoint, early_stop],
    verbose=1
)