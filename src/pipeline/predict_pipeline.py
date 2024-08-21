import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Define numerical columns
numerical_columns = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment"
]

# Define categorical columns
categorical_columns = [
    "Store", "Holiday_Flag"
]

# Define all columns
all_columns = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price", 
    "CPI", "Unemployment", "Year", "Month", "Day", "Weekday"
]

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "best_model.pkl")
            preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Store: int, Holiday_Flag: int, Temperature: float, 
                 Fuel_Price: float, CPI: float, Unemployment: float, 
                 Year: int, Month: int, Day: int, Weekday: int):
        self.Store = Store
        self.Holiday_Flag = Holiday_Flag
        self.Temperature = Temperature
        self.Fuel_Price = Fuel_Price
        self.CPI = CPI
        self.Unemployment = Unemployment
        self.Year = Year
        self.Month = Month
        self.Day = Day
        self.Weekday = Weekday

    def get_data_as_data_frame(self):
        try:
            # Build dictionary for all features
            custom_data_input_dict = {
                "Store": [self.Store],
                "Holiday_Flag": [self.Holiday_Flag],
                "Temperature": [self.Temperature],
                "Fuel_Price": [self.Fuel_Price],
                "CPI": [self.CPI],
                "Unemployment": [self.Unemployment],
                "Year": [self.Year],
                "Month": [self.Month],
                "Day": [self.Day],
                "Weekday": [self.Weekday]
            }

            # Convert to DataFrame
            return pd.DataFrame(custom_data_input_dict, columns=all_columns)
        
        except Exception as e:
            raise CustomException(e, sys)