import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            
            # Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data transformation completed")
            
            # Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training completed")
            
            logging.info("Training pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.initiate_training_pipeline()


