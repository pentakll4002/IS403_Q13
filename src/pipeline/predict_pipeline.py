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
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            processed_features = self._prepare_features(features, preprocessor)
            preds = model.predict(processed_features)
            probas = model.predict_proba(processed_features)[:, 1]
            return preds, probas
        except Exception as e:
            raise CustomException(e, sys)
    
    def _prepare_features(self, df, preprocessor):
        try:
            df = df.copy()
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
            df['Transaction_Year'] = df['Transaction Date'].dt.year
            df['Transaction_Month'] = df['Transaction Date'].dt.month
            df['Transaction_Day'] = df['Transaction Date'].dt.day
            df['Transaction_Hour'] = df['Transaction Date'].dt.hour
            df['Transaction_Minute'] = df['Transaction Date'].dt.minute
            df['Transaction_Second'] = df['Transaction Date'].dt.second
            df['Transaction_DayOfWeek'] = df['Transaction Date'].dt.dayofweek
            df['Is_Weekend'] = df['Transaction_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df.drop(columns=["Transaction Date"], inplace=True)
            return preprocessor.transform(df)
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
