import numpy as np
import tensorflow as tf
import joblib

class StockPredictor:
    def __init__(self, model_path='best_gru.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load('scaler.pkl')
        self.look_back = 60
        
    def rolling_predict(self, test_data):
        scaled_test = self.scaler.transform(test_data[['close']])
        
        predictions = []
        for i in range(len(test_data) - self.look_back):
            X = scaled_test[i:i+self.look_back].reshape(1, self.look_back, 1)
            pred = self.model.predict(X, verbose=0)[0][0]
            predictions.append(pred)
        
        if predictions:
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            full_pred = np.full(len(test_data), np.nan)
            full_pred[self.look_back:] = predictions.flatten()
        else:
            full_pred = np.full(len(test_data), np.nan)
        
        return full_pred