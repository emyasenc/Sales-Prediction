import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_pipeline():
    try:
        # Data Ingestion
        logging.info("Starting data ingestion process...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        logging.info("Starting data transformation process...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model Training
        logging.info("Starting model training process...")
        model_trainer = ModelTrainer()
        model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Model Evaluation
        logging.info("Evaluating models...")
        for model_name, model_info in model_report.items():
            if 'best_model' in model_info:
                model = model_info['best_model']
                X_train, y_train, X_test, y_test = (
                    train_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, :-1],
                    test_arr[:, -1]
                )
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                metrics = {
                    "mean_absolute_error": mean_absolute_error(y_test, y_test_pred),
                    "mean_squared_error": mean_squared_error(y_test, y_test_pred),
                    "r2_score": r2_score(y_test, y_test_pred)
                }

                logging.info(f"Metrics for {model_name}: {metrics}")
            else:
                logging.warning(f"Best model not available for {model_name}")

        return model_report

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline()