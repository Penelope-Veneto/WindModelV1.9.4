import numpy as np
import lightgbm as lgb
from src.metrics import pointwise_accuracy

class Trainer:
    def __init__(self, model_params: dict, capacity_kw: float):
        # 提取原生接口需要的参数
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': model_params.get('learning_rate', 0.01),
            'num_leaves': model_params.get('num_leaves', 31),
            'feature_fraction': model_params.get('feature_fraction', 0.8),
            'bagging_fraction': model_params.get('bagging_fraction', 0.8),
            'random_state': model_params.get('random_state', 42)
        }
        self.num_boost_round = model_params.get('n_estimators', 1000)
        self.early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
        self.capacity_kw = capacity_kw

    def train(self, X_train, y_train, X_valid, y_valid):
        # 使用 LightGBM 原生 Dataset，不走 sklearn 接口，彻底绕过报错
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            ]
        )
        return model

    def evaluate(self, model, X_test, y_test):
        # 原生模型的 predict 是一样的
        y_pred = model.predict(X_test)
        metrics_dict = pointwise_accuracy(y_test, y_pred, self.capacity_kw)
        return metrics_dict["R2"], metrics_dict["MAE"], metrics_dict["Accuracy"]