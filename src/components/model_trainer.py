import os 
import sys 

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..exception import CustomException
from ..logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Define models to train
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'CatBoost': CatBoostRegressor(verbose=False, random_seed=42)
            }
            
            model_report = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate R2 scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Calculate additional metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                model_report[model_name] = test_model_score
                
                logging.info(f"{model_name} - Train R2 Score: {train_model_score:.4f}")
                logging.info(f"{model_name} - Test R2 Score: {test_model_score:.4f}")
                logging.info(f"{model_name} - Train MSE: {train_mse:.4f}")
                logging.info(f"{model_name} - Test MSE: {test_mse:.4f}")
                logging.info(f"{model_name} - Train MAE: {train_mae:.4f}")
                logging.info(f"{model_name} - Test MAE: {test_mae:.4f}")
            
            # Find the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")
            
            # Save the best model
            import joblib
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            
            logging.info("Model training completed successfully")
            
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)