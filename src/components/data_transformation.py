import os 
import sys 
import pandas as pd
import numpy as np
import ipaddress

from ..logger import logging
from ..exception import CustomException

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import category_encoders as ce

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

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
        """
        Convert IP Address to integer format
        """
        try:
            logging.info("Starting IP address conversion")
            
            def ip_to_int(ip):
                try:
                    return int(ipaddress.ip_address(ip))
                except:
                    return None
            
            df['IP_Int'] = df['IP Address'].apply(ip_to_int)
            df.drop(columns=["IP Address"], inplace=True)
            
            logging.info("IP address conversion completed successfully")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
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
                "Billing Address"
            ]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            logging.info("Column dropping completed successfully")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_categorical_features(self):
        """
        Return list of categorical features
        """
        return ["Payment Method", "Product Category", "Customer Location", "Device Used"]
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation pipeline
        """
        try:
            logging.info("Starting data transformer object creation")
            
            categorical_features = self.get_categorical_features()
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
                    ('num', StandardScaler(), ['Transaction Amount', 'Quantity', 'Customer Age', 
                                               'Account Age Days', 'Transaction_Year',
                                               'Transaction_Month', 'Transaction_Day', 'Transaction_Hour',
                                               'Transaction_Minute', 'Transaction_Second', 'Transaction_DayOfWeek',
                                               'Is_Weekend', 'IP_Int'])
                ],
                remainder='passthrough'
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
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Apply transformations
            train_df = self.extract_datetime_features(train_df)
            train_df = self.convert_ip_to_int(train_df)
            train_df = self.drop_unnecessary_columns(train_df)
            
            test_df = self.extract_datetime_features(test_df)
            test_df = self.convert_ip_to_int(test_df)
            test_df = self.drop_unnecessary_columns(test_df)
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'Transaction Amount'
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)