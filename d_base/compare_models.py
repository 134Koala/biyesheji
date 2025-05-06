from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class ModelComparator:
    @staticmethod
    def compare_models(X_train, y_train, X_test, y_test):
        # 对比随机森林
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        rf_pred = rf.predict(X_test.reshape(X_test.shape[0], -1))
        
        # 对比XGBoost
        xgb = XGBRegressor(n_estimators=100)
        xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        xgb_pred = xgb.predict(X_test.reshape(X_test.shape[0], -1))
        
        return {
            'RandomForest': mean_squared_error(y_test, rf_pred),
            'XGBoost': mean_squared_error(y_test, xgb_pred)
        }