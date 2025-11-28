import os 
import sys 

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from ..exception import CustomException
from ..logger import logging
from ..utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42
        )

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training for fraud classification")
            y_train = np.ravel(y_train)
            y_test = np.ravel(y_test)

            self.model.fit(X_train, y_train)
            logging.info("Model fitting completed")

            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            save_object(self.config.trained_model_file_path, self.model)
            logging.info(f"Persisted trained classifier to {self.config.trained_model_file_path}")

            return {
                "metrics": metrics,
                "model_path": self.config.trained_model_file_path,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
        except Exception as e:
            raise CustomException(e, sys)