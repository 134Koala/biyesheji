import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Bidirectional, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

class GRUModel:
    def __init__(self, look_back=60):
        self.model = None
        self.look_back = look_back
        
    def build_model(self, units=64, dropout=0.3):
        model = Sequential([
            Bidirectional(GRU(units, return_sequences=True), 
                         input_shape=(self.look_back, 1)),
            Dropout(dropout),
            GRU(units//2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):  # 新增验证集参数
        best_score = float('inf')
        best_build_params = {}
        best_fit_params = {}
        
        # 遍历参数组合（示例参数）
        # 在GRUModel类中修改参数组合
        param_combinations = [
            {'build_params': {'units': 64, 'dropout': 0.2}, 'fit_params': {'batch_size': 32}},
            {'build_params': {'units': 128, 'dropout': 0.3}, 'fit_params': {'batch_size': 64}},
            # 可扩展更多组合
        ]
        
        for combo in param_combinations:
            # 解包参数
            build_params = combo['build_params']
            fit_params = combo['fit_params']
            
            # 构建模型（仅传递结构参数）
            model = self.build_model(**build_params)
            
            # 训练模型（传递训练参数）
            history = model.fit(
                X_train, y_train,
                epochs=50,
                **fit_params,  # 传入batch_size等训练参数
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # 评估验证损失
            val_loss = min(history.history['val_loss'])
            if val_loss < best_score:
                best_score = val_loss
                best_build_params = build_params
                best_fit_params = fit_params
        
        # 使用最佳参数重建模型
        self.model = self.build_model(**best_build_params)
        return {**best_build_params,**best_fit_params}  # 返回完整参数报告
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            self.model = self.build_model()
            
        # 在train方法中加强早停控制
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                min_delta=0.0001,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_gru.h5',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        return history