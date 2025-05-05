# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Dropout
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import GRU, Dense, Dropout
from keras.api._v2.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.api._v2.keras.layers import Bidirectional, Attention, RepeatVector, TimeDistributed
from keras.api._v2.keras.optimizers import Adam
import tensorflow as tf

def build_model(input_shape, forecast_steps):
    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        TimeDistributed(Dense(32, activation='relu')),
        GRU(32),
        Dense(forecast_steps)
    ])
    
    optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='huber')
    return model

def train_model():
    # 获取数据
    from data_get import get_stock_data
    X_train, y_train, X_test, y_test, scaler, forecast_steps = get_stock_data(
        "000001", "2010-01-01", "2020-12-31"
    )
    
    # 构建模型
    model = build_model((X_train.shape[1], 1), forecast_steps)
    
    # 回调函数
    callbacks = [
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # 训练
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=False
    )
    return model, history