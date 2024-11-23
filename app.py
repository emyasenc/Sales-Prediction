from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('notebook/best_model.pkl')

# Feature engineering and preprocessing functions
def preprocess_input(data):
    # Create a DataFrame for input data
    df_input = pd.DataFrame([data])

    # Encode categorical variables
    df_input['Holiday_Flag'] = df_input['Holiday_Flag'].astype('category')
    df_input['Store'] = df_input['Store'].astype('category')

    # Extract date-based features (if needed)
    # Simulate date-based features if needed
    df_input['Year'] = df_input['Year']
    df_input['Month'] = df_input['Month']
    df_input['Day'] = df_input['Day']
    df_input['Weekday'] = df_input['Weekday']

    # Convert categorical columns to integers
    df_input['Store'] = df_input['Store'].cat.codes
    df_input['Holiday_Flag'] = df_input['Holiday_Flag'].cat.codes

    # Feature Scaling
    numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    scaler = StandardScaler()
    
    # Apply scaling only to the numerical features
    df_input[numerical_features] = scaler.fit_transform(df_input[numerical_features])

    # Handle outliers
    z_scores = stats.zscore(df_input['Temperature'])
    df_input['Temperature'] = np.where(z_scores > 3, np.nan, df_input['Temperature'])
    df_input['Temperature'] = df_input['Temperature'].fillna(df_input['Temperature'].median())
    
    z_scores = stats.zscore(df_input['Fuel_Price'])
    df_input['Fuel_Price'] = np.where(z_scores > 3, np.nan, df_input['Fuel_Price'])
    df_input['Fuel_Price'] = df_input['Fuel_Price'].fillna(df_input['Fuel_Price'].median())

    # Return preprocessed DataFrame
    return df_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['POST'])
def predict_datapoint():
    # Extract input data from the form
    user_input = {
        'Store': request.form['Store'],
        'Holiday_Flag': request.form['Holiday_Flag'],
        'Temperature': request.form['Temperature'],
        'Fuel_Price': request.form['Fuel_Price'],
        'CPI': request.form['CPI'],
        'Unemployment': request.form['Unemployment'],
        'Year': request.form['Year'],
        'Month': request.form['Month'],
        'Day': request.form['Day'],
        'Weekday': request.form['Weekday']
    }

    # Preprocess the input data
    preprocessed_data = preprocess_input(user_input)

    # Make prediction using the trained model
    prediction = model.predict(preprocessed_data)

    # Return the prediction result to the user
    return render_template('index.html', results=f'Predicted Sales: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
       