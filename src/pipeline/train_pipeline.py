import os
import sys
import tempfile
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
    def __init__(self, tracking_uri="file:./mlruns", experiment_name="FraudDetection"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def initiate_training_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            data_ingestion = DataIngestion()
            data_transformation = DataTransformation()
            model_trainer = ModelTrainer()

            logging.info("Starting data ingestion")
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")

            logging.info("Starting data transformation")
            (
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor_path,
            ) = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data transformation completed")

            logging.info("Starting model training")
            trainer_output = model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )
            metrics = trainer_output["metrics"]
            logging.info(f"Classification metrics: {metrics}")

            with mlflow.start_run(run_name="fraud_classification_by_rf"):
                mlflow.set_tag("model_type", "XGBoost")
                mlflow.log_params(model_trainer.model.get_params())
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, float(value))

                if preprocessor_path and os.path.exists(preprocessor_path):
                    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

                mlflow.sklearn.log_model(
                    sk_model=model_trainer.model,
                    artifact_path="model",
                )

                roc_path = self._save_roc_curve(
                    trainer_output["y_test"], trainer_output["y_proba"]
                )
                if roc_path:
                    mlflow.log_artifact(roc_path, artifact_path="plots")

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e

    def _save_roc_curve(self, y_true, y_proba):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            tmp_dir = tempfile.mkdtemp()
            plot_path = os.path.join(tmp_dir, "roc_curve.png")
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.plot([0, 1], [0, 1], "r--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as plot_error:
            logging.error(f"Failed to save ROC curve: {plot_error}")
            return None

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.initiate_training_pipeline()


































