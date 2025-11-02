import sys
import pandas as pd 
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
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        

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
        ip_address: str
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
        self.ip_address = ip_address

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
                "Transaction Date": [self.transaction_date],
                "IP Address": [self.ip_address]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
