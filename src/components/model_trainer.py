import os 
import sys 

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from ..exception import CustomException
from ..logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }