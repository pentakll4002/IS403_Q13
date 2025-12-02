import os 
import sys 
import pandas as pd
import numpy as np
import ipaddress

from ..logger import logging
from ..exception import CustomException

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

from ..utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.high_card_categorical = ["Customer Location"]
        self.small_categorical = ["Payment Method", "Product Category", "Device Used"]

    def _engineer_features(
        self,
        df: pd.DataFrame,
        customer_freq: dict | None = None,
        avg_amount_per_cat: pd.Series | None = None,
        is_train: bool = True,
    ):
        try:
            df = df.copy()

            df = df[df["Customer Age"] > 0]

            df["Address_Mismatch"] = (
                df["Shipping Address"] != df["Billing Address"]
            ).astype(int)

            df["Transaction Date"] = pd.to_datetime(
                df["Transaction Date"], errors="coerce"
            )
            df["Transaction_Weekday"] = df["Transaction Date"].dt.dayofweek
            df["Transaction_Hour"] = df["Transaction Date"].dt.hour

            if is_train:
                customer_freq = df["Customer ID"].value_counts().to_dict()
            if customer_freq is not None:
                df["Customer_Frequency"] = df["Customer ID"].map(customer_freq).fillna(
                    0
                )

            if is_train:
                avg_amount_per_cat = (
                    df.groupby("Product Category")["Transaction Amount"].mean()
                )
            if avg_amount_per_cat is not None:
                df["Amount_Higher_Than_Average"] = (
                    df["Transaction Amount"]
                    > df["Product Category"].map(avg_amount_per_cat)
                ).astype(int)

            cols_to_drop = [
                "Transaction ID",
                "Customer ID",
                "Shipping Address",
                "Billing Address",
                "IP Address",
                "Transaction Date",
            ]
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

            return df, customer_freq, avg_amount_per_cat

        except Exception as e:
            raise CustomException(e, sys)
    
    def get_numeric_features(self, df):
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["Is Fraudulent"]]
        return numeric_cols
    
    def get_data_transformer_object(self, df):
        """
        This function is responsible for data transformation pipeline
        """
        try:
            logging.info("Starting data transformer object creation (linear + tree)")

            # Numeric features (same cho cả 2 preprocessor)
            numeric_features = self.get_numeric_features(df)

            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            cat_low = [c for c in categorical_cols if df[c].nunique() < 50]
            cat_high = [c for c in categorical_cols if df[c].nunique() >= 50]

            preprocessor_linear = ColumnTransformer(
                transformers=[
                    (
                        "target",
                        TargetEncoder(min_samples_leaf=10, smoothing=30),
                        cat_high,
                    ),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_low),
                    ("scale", StandardScaler(), numeric_features),
                ],
                remainder="drop",
            )

            preprocessor_tree = ColumnTransformer(
                transformers=[
                    (
                        "target",
                        TargetEncoder(min_samples_leaf=10, smoothing=30),
                        cat_high,
                    ),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        cat_low,
                    ),
                ],
                remainder="passthrough",
            )
            self.preprocessor_linear = preprocessor_linear
            self.preprocessor_tree = preprocessor_tree

            logging.info(
            
                "Data transformer objects created successfully "
                "(preprocessor_linear + preprocessor_tree). "
                "Returning preprocessor_linear as default."
            )
            return preprocessor_tree
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiate data transformation process
        """
        try:
            logging.info("Starting data transformation process")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            logging.info("Applying feature engineering like notebook (train)")
            train_df, customer_freq, avg_amount_per_cat = self._engineer_features(
                train_df, is_train=True
            )

            logging.info("Applying feature engineering like notebook (test)")
            test_df, _, _ = self._engineer_features(
                test_df,
                customer_freq=customer_freq,
                avg_amount_per_cat=avg_amount_per_cat,
                is_train=False,
            )

            logging.info(
                f"Columns in train_df before preprocessing object: {train_df.columns.tolist()}"
            )
            logging.info(
                f"Columns in test_df before preprocessing object: {test_df.columns.tolist()}"
            )
            
            preprocessing_obj = self.get_data_transformer_object(train_df)
            
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
            
            logging.info("Data transformation (feature engineering) completed successfully")
            logging.info("Returning raw DataFrames and preprocessor object for ImbPipeline")
            
            return (
                train_df,
                test_df,
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)