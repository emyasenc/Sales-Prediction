import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Define numerical columns
numerical_columns = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment", 
    "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5", 
    "Rolling_Mean", "Rolling_Std", "Days_to_Holiday",
    "Day_of_Week", "Week_of_Year", "Quarter", "Is_Mid_Month"
]

# Define categorical columns
categorical_columns = [
    "Store", "Holiday_Flag", "Is_Holiday"
]

# Define all columns
all_columns = [
    "Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", 
    "CPI", "Unemployment", "Month", "Lag_1", "Lag_2", "Lag_3", "Lag_4", 
    "Lag_5", "Rolling_Mean", "Rolling_Std", "Days_to_Holiday", "Is_Holiday", 
    "Day_of_Week", "Week_of_Year", "Quarter", "Is_Mid_Month"
]

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "model.pkl")
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
    def __init__(self, Store: int, Date: str, Holiday_Flag: int, Temperature: float, 
                 Fuel_Price: float, CPI: float, Unemployment: float, Month: int, 
                 Lag_1: float, Lag_2: float, Lag_3: float, Lag_4: float, Lag_5: float, 
                 Rolling_Mean: float, Rolling_Std: float, Days_to_Holiday: float, 
                 Is_Holiday: int, Day_of_Week: int, Week_of_Year: int, Quarter: int, 
                 Is_Mid_Month: int):
        self.Store = Store
        self.Date = Date
        self.Holiday_Flag = Holiday_Flag
        self.Temperature = Temperature
        self.Fuel_Price = Fuel_Price
        self.CPI = CPI
        self.Unemployment = Unemployment
        self.Month = Month
        self.Lag_1 = Lag_1
        self.Lag_2 = Lag_2
        self.Lag_3 = Lag_3
        self.Lag_4 = Lag_4
        self.Lag_5 = Lag_5
        self.Rolling_Mean = Rolling_Mean
        self.Rolling_Std = Rolling_Std
        self.Days_to_Holiday = Days_to_Holiday
        self.Is_Holiday = Is_Holiday
        self.Day_of_Week = Day_of_Week
        self.Week_of_Year = Week_of_Year
        self.Quarter = Quarter
        self.Is_Mid_Month = Is_Mid_Month

    def get_data_as_data_frame(self):
        try:
            # Build dictionary for all features
            custom_data_input_dict = {
                "Store": [self.Store],
                "Date": [self.Date],
                "Holiday_Flag": [self.Holiday_Flag],
                "Temperature": [self.Temperature],
                "Fuel_Price": [self.Fuel_Price],
                "CPI": [self.CPI],
                "Unemployment": [self.Unemployment],
                "Month": [self.Month],
                "Lag_1": [self.Lag_1],
                "Lag_2": [self.Lag_2],
                "Lag_3": [self.Lag_3],
                "Lag_4": [self.Lag_4],
                "Lag_5": [self.Lag_5],
                "Rolling_Mean": [self.Rolling_Mean],
                "Rolling_Std": [self.Rolling_Std],
                "Days_to_Holiday": [self.Days_to_Holiday],
                "Is_Holiday": [self.Is_Holiday],
                "Day_of_Week": [self.Day_of_Week],
                "Week_of_Year": [self.Week_of_Year],
                "Quarter": [self.Quarter],
                "Is_Mid_Month": [self.Is_Mid_Month]
            }

            # Convert to DataFrame
            return pd.DataFrame(custom_data_input_dict, columns=all_columns)
        
        except Exception as e:
            raise CustomException(e, sys)