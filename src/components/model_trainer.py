import os 
import sys 
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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
            max_features='sqrt'
        )

    def initiate_model_trainer(self, train_df, test_df, preprocessor):
        """
        Initiate model training with ImbPipeline
        Args:
            train_df: Training DataFrame with features and target
            test_df: Test DataFrame with features and target
            preprocessor: ColumnTransformer object (not fitted)
        """
        try:
            logging.info("Starting model training with ImbPipeline for fraud classification")
            
            target_column_name = 'Is Fraudulent'
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            
            logging.info(f"Train data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            
            logging.info(
                "Creating ImbPipeline with preprocessor, SMOTE, and LogisticRegression"
            )
            self.pipeline = ImbPipeline([
                ("prep", preprocessor),
                ("smote", SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)),
                ("clf", self.model)
            ])
            
            # Fit pipeline on raw training data
            # Pipeline will automatically: preprocess -> SMOTE -> train model
            logging.info("Fitting ImbPipeline on training data")
            self.pipeline.fit(X_train, y_train)
            logging.info("ImbPipeline fitting completed")
            
            # Predict on test data (pipeline automatically preprocesses test data, no SMOTE)
            logging.info("Predicting on test data")
            y_pred = self.pipeline.predict(X_test)
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            save_object(self.config.trained_model_file_path, self.pipeline)
            logging.info(f"Persisted trained pipeline to {self.config.trained_model_file_path}")

            return {
                "metrics": metrics,
                "model_path": self.config.trained_model_file_path,
                "y_test": y_test.values,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
        except Exception as e:
            raise CustomException(e, sys)