import os 
import sys 

from ..logger import logging
from ..exception import CustomException

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """
    This code responsible for ingestion (Extract)
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")

        try:
            dataset_paths = [
                "data/Fraudulent_E-Commerce_Transaction_Data.csv",
                "data/Fraudulent_E-Commerce_Transaction_Data_2.csv"
            ]

            dataframes = []
            for path in dataset_paths:
                if not os.path.exists(path):
                    raise CustomException(f"Dataset not found at {path}", sys)
                logging.info(f"Reading dataset: {path}")
                dataframes.append(pd.read_csv(path))

            df = pd.concat(dataframes, ignore_index=True)
            logging.info("Successfully concatenated fraud datasets")
            if "IP Address" in df.columns:
                df.drop(columns=["IP Address"], inplace=True)
                logging.info("Dropped IP Address column as requested")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Is Fraudulent"])

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of the dataset is completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e: 
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()