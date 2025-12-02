import sys
import pandas as pd
import numpy as np
from datetime import datetime

from ..exception import CustomException
from ..logger import logging
from ..utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            pipeline_path = 'artifacts/model.pkl'
            pipeline = load_object(file_path=pipeline_path)
            prepared_features = self._prepare_features(features)
            # Pipeline automatically handles preprocessing (no SMOTE for prediction)
            preds = pipeline.predict(prepared_features)
            probas = pipeline.predict_proba(prepared_features)[:, 1]
            return preds, probas
        except Exception as e:
            raise CustomException(e, sys)
    
    def _prepare_features(self, df):
        """
        Prepare features by applying same feature engineering as in DataTransformation:
        - lọc age > 0
        - Address_Mismatch
        - Transaction_Weekday, Transaction_Hour
        - Customer_Frequency, Amount_Higher_Than_Average sẽ được model xử lý theo
          thống kê đã học trên train (mapping được "đóng" trong preprocessor/pipeline)
        - drop các cột ID / address / IP / Transaction Date
        Pipeline will handle preprocessing (encoding, scaling)
        """
        try:
            df = df.copy()
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Lọc Customer Age > 0 giống notebook
            df = df[df["Customer Age"] > 0]

            # Address mismatch
            df["Address_Mismatch"] = (
                df["Shipping Address"] != df["Billing Address"]
            ).astype(int)

            # Datetime -> weekday & hour giống notebook
            df["Transaction Date"] = pd.to_datetime(
                df["Transaction Date"], errors="coerce"
            )
            df["Transaction_Weekday"] = df["Transaction Date"].dt.dayofweek
            df["Transaction_Hour"] = df["Transaction Date"].dt.hour
            df.drop(columns=["Transaction Date"], inplace=True)
            
            # Drop unnecessary columns (same as data_transformation)
            cols_to_drop = [
                "Transaction ID",
                "Customer ID",
                "Shipping Address",
                "Billing Address",
                "IP Address",
            ]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            return df
        except Exception as prep_error:
            raise CustomException(prep_error, sys)
        

class CustomData:
    def __init__(
        self,
        transaction_amount: float,
        quantity: int,
        customer_age: int,
        account_age_days: int,
        payment_method: str,
        product_category: str,
        customer_location: str,
        device_used: str,
        transaction_date: str,
    ):
        self.transaction_amount = transaction_amount
        self.quantity = quantity
        self.customer_age = customer_age
        self.account_age_days = account_age_days
        self.payment_method = payment_method
        self.product_category = product_category
        self.customer_location = customer_location
        self.device_used = device_used
        self.transaction_date = transaction_date

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Transaction Amount": [self.transaction_amount],
                "Quantity": [self.quantity],
                "Customer Age": [self.customer_age],
                "Account Age Days": [self.account_age_days],
                "Payment Method": [self.payment_method],
                "Product Category": [self.product_category],
                "Customer Location": [self.customer_location],
                "Device Used": [self.device_used],
                "Transaction Date": [self.transaction_date]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
