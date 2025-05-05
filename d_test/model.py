# 文件2：model.py（新增数据分割逻辑）
from data_processor import preprocess_data, create_dataset
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def prepare_datasets(symbol, start_date, end_date):
    """统一准备数据集"""
    full_data, scaler = preprocess_data(symbol, start_date, end_date)
    
    # 按时间顺序分割数据集
    train_size = int(len(full_data) * 0.8)
    train_data = full_data.iloc[:train_size]
    test_data = full_data.iloc[train_size:]
    
    # 生成监督数据集
    X_train, y_train, scaler = create_dataset(train_data, scaler, look_back=60, forecast_steps=1)
    X_test, y_test, _ = create_dataset(test_data, scaler, look_back=60, forecast_steps=1)
    
    return (X_train, y_train), (X_test, y_test), scaler

def build_model(input_shape, forecast_steps):
    """模型构建保持不变"""
    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        TimeDistributed(Dense(32, activation='relu')),
        GRU(32),
        Dense(forecast_steps)
    ])
    model.compile(optimizer='adam', loss='huber')
    return model

def train_model():
    """训练流程"""
    # 统一获取数据
    (X_train, y_train), (X_test, y_test), scaler = prepare_datasets(
        "000001", "2010-01-01", "2020-12-31"
    )
    
    # 模型构建和训练
    model = build_model((X_train.shape[1], 1), y_train.shape[1])
    callbacks = [
        ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True),
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        shuffle=False
    )
    
    # 返回测试集用于后续预测
    return model, X_test, y_test, scaler