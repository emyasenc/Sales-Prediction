# Sales Prediction Using Walmart Data

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies](#technologies)
- [Data Description](#data-description)
- [Data Ingestion](#data-ingestion)
- [Data Transformation](#data-transformation)
- [Model Training](#model-training)
- [Prediction Pipeline](#prediction-pipeline)
- [How to Run](#how-to-run)
- [License](#license)

## Project Overview

This project aims to build a predictive model to forecast weekly sales for Walmart stores using historical sales data. The model employs various regression techniques, and the data preprocessing steps include feature engineering and transformation to enhance the model's predictive performance.

## Technologies

Python 3.x
pandas
NumPy
scikit-learn
XGBoost
CatBoost
LightGBM
Matplotlib
Logging
Dataclasses

## Data Description

The dataset consists of historical sales data for Walmart stores, which includes features like:

Store: The store identifier.
Holiday_Flag: Indicates whether the week includes a holiday.
Temperature: Average temperature for the week.
Fuel_Price: Average fuel price for the week.
CPI: Consumer Price Index for the week.
Unemployment: Unemployment rate for the week.
Weekly_Sales: The target variable, representing total sales for the week.

## Components

Data Ingestion: Responsible for loading and managing data.

data_ingestion.py: Loads preprocessed train and test datasets.
Data Transformation: Handles feature engineering and preprocessing.

data_transformation.py: Implements pipelines for numerical and categorical features, including scaling and polynomial feature generation.
Model Trainer: Trains multiple regression models and selects the best performing one.

model_trainer.py: Utilizes various regression models, evaluates their performance using cross-validation, and saves the best model.
Prediction Pipeline: Makes predictions based on the trained model and preprocessor.

predict_pipeline.py: Loads the model and preprocessor to transform incoming feature data for prediction.

## How to Run

1. Clone the repository:
   
   git clone <repository-url>
   cd sales_prediction_walmart
   
2. Install the required packages:
   
   pip install -r requirements.txt
   
3. 
