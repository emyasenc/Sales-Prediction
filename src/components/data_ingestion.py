import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

# Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'preprocessed', 'train_data.csv')
    test_data_path: str = os.path.join('data', 'preprocessed', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the preprocessed train and test data
            logging.info("Loading the preprocessed train and test datasets")
            train_df = pd.read_csv(self.ingestion_config.train_data_path)
            test_df = pd.read_csv(self.ingestion_config.test_data_path)

            logging.info("Successfully loaded the train and test datasets")

            # Return the paths (in case other components need to reload the data)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))