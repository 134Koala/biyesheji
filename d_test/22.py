# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Dropout
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import GRU, Dense, Dropout
from keras.api._v2.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.api._v2.keras.layers import Bidirectional, Attention, RepeatVector, TimeDistributed
import data_get as dg

model = Sequential()
# 双向GRU捕捉更复杂的时间模式
model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(dg.X_train.shape[1], 1)))
model.add(Dropout(0.3))

# 添加注意力层
model.add(GRU(64, return_sequences=True))
model.add(Attention()( [model.output, model.output] ))  # 自注意力机制

# 时间分布全连接层处理序列
model.add(TimeDistributed(Dense(32, activation='relu')))
model.add(GRU(32))

# 多步预测输出
model.add(Dense(dg.FORECAST_STEPS))  # 输出多个预测步长

# model.compile(
#     optimizer='adam',
#     loss='mean_squared_error',
#     metrics=['mae']  # 监控平均绝对误差
# )
# 使用自定义学习率

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer, loss='huber_loss')  # 使用对异常值更鲁棒的损失函数

# 回调函数配置 [5](@ref)
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# early_stop = EarlyStopping(patience=10, restore_best_weights=True)
# 增加epochs并改进早停策略
early_stop = EarlyStopping(
    monitor='val_mae',
    patience=20,
    mode='min',
    restore_best_weights=True
)

# 添加学习率调度器
from keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# 模型训练
history = model.fit(
    dg.X_train, dg.y_train,
    epochs=200,  # 增加训练轮次
    batch_size=64,  # 适当增大batch_size
    validation_data=(dg.X_test, dg.y_test),
    callbacks=[checkpoint, early_stop, lr_scheduler],
    shuffle=False
)
# history = model.fit(
#     dg.X_train, dg.y_train,
#     epochs=100,
#     batch_size=32,
#     validation_data=(dg.X_test, dg.y_test),
#     callbacks=[checkpoint, early_stop],
#     shuffle=False,  # 添加时间序列关键参数
#     verbose=1
# )