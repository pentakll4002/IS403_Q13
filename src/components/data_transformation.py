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

    def extract_datetime_features(self, df):
        """
        Extract datetime features from Transaction Date column
        """
        try:
            logging.info("Starting datetime feature extraction")
            
            # Convert Transaction Date to datetime
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
            
            # Extract datetime features
            df['Transaction_Year'] = df['Transaction Date'].dt.year
            df['Transaction_Month'] = df['Transaction Date'].dt.month
            df['Transaction_Day'] = df['Transaction Date'].dt.day
            df['Transaction_Hour'] = df['Transaction Date'].dt.hour
            df['Transaction_Minute'] = df['Transaction Date'].dt.minute
            df['Transaction_Second'] = df['Transaction Date'].dt.second
            df['Transaction_DayOfWeek'] = df['Transaction Date'].dt.dayofweek  
            df['Is_Weekend'] = df['Transaction_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Drop original Transaction Date column
            df.drop(columns=["Transaction Date"], inplace=True)
            
            logging.info("Datetime feature extraction completed successfully")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def convert_ip_to_int(self, df):
        return df
    
    def drop_unnecessary_columns(self, df):
        """
        Drop columns that are not needed for model training
        """
        try:
            logging.info("Starting column dropping")
            
            cols_to_drop = [
                "Transaction ID",
                "Customer ID",
                "Shipping Address",
                "Billing Address",
                "IP Address",
            ]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            logging.info("Column dropping completed successfully")
            return df
            
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
            logging.info("Starting data transformer object creation")
            
            numeric_features = self.get_numeric_features(df)
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("target", TargetEncoder(min_samples_leaf=10, smoothing=30), self.high_card_categorical),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.small_categorical),
                    ("scale", StandardScaler(), numeric_features),
                ],
                remainder="drop"
            )
            
            logging.info("Data transformer object created successfully")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiate data transformation process
        """
        try:
            logging.info("Starting data transformation process")
            
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Strip whitespace from column names to avoid KeyError issues
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Apply transformations
            train_df = self.extract_datetime_features(train_df)
            train_df = self.convert_ip_to_int(train_df)
            train_df = self.drop_unnecessary_columns(train_df)
            
            test_df = self.extract_datetime_features(test_df)
            test_df = self.convert_ip_to_int(test_df)
            test_df = self.drop_unnecessary_columns(test_df)
            
            # Only drop the original 'Transaction Hour' (with spaces), not the engineered 'Transaction_Hour'
            for df in [train_df, test_df]:
                to_drop = [col for col in df.columns if col.strip() == 'Transaction Hour']
                if to_drop:
                    df.drop(columns=to_drop, inplace=True)

            logging.info(f"Columns in train_df before splitting: {train_df.columns.tolist()}")
            logging.info(f"Columns in test_df before splitting: {test_df.columns.tolist()}")
            
            # Get preprocessing object (not fitted yet)
            preprocessing_obj = self.get_data_transformer_object(train_df)
            
            # Save preprocessor object for later use
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
            
            logging.info("Data transformation (feature engineering) completed successfully")
            logging.info("Returning raw DataFrames and preprocessor object for ImbPipeline")
            
            # Return raw DataFrames and preprocessor object (not fitted)
            # Model trainer will create ImbPipeline and fit it
            return (
                train_df,
                test_df,
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)