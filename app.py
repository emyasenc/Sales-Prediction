from flask import Flask, request, render_template
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app=Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # Collect form data with default values and type conversion
            store = request.form.get('Store')
            holiday_flag = request.form.get('Holiday_Flag')
            temperature = request.form.get('Temperature')
            fuel_price = request.form.get('Fuel_Price')
            cpi = request.form.get('CPI')
            unemployment = request.form.get('Unemployment')
            year = request.form.get('Year')
            month = request.form.get('Month')
            day = request.form.get('Day')
            weekday = request.form.get('Weekday')

            # Debug print statements to check form values
            print("Form Data:", {
                'Store': store,
                'Holiday_Flag': holiday_flag,
                'Temperature': temperature,
                'Fuel_Price': fuel_price,
                'CPI': cpi,
                'Unemployment': unemployment,
                'Year': year,
                'Month': month,
                'Day': day,
                'Weekday': weekday
            })

            # Convert to appropriate types with default values if necessary
            store = int(store) if store else 0
            holiday_flag = int(holiday_flag) if holiday_flag else 0
            temperature = float(temperature) if temperature else 0.0
            fuel_price = float(fuel_price) if fuel_price else 0.0
            cpi = float(cpi) if cpi else 0.0
            unemployment = float(unemployment) if unemployment else 0.0
            year = int(year) if year else 0
            month = int(month) if month else 0
            day = int(day) if day else 0
            weekday = int(weekday) if weekday else 0

            # Debug print statements to check converted values
            print("Converted Data:", {
                'Store': store,
                'Holiday_Flag': holiday_flag,
                'Temperature': temperature,
                'Fuel_Price': fuel_price,
                'CPI': cpi,
                'Unemployment': unemployment,
                'Year': year,
                'Month': month,
                'Day': day,
                'Weekday': weekday
            })
            
            # Create CustomData instance
            data = CustomData(
                Store=store,
                Holiday_Flag=holiday_flag,
                Temperature=temperature,
                Fuel_Price=fuel_price,
                CPI=cpi,
                Unemployment=unemployment,
                Year=year,
                Month=month,
                Day=day,
                Weekday=weekday
            )

            # Prepare the DataFrame for prediction
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame for Prediction:", pred_df)

            # Predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Display results
            print("Prediction results:", results)
            return render_template('index.html', results=results[0])

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', results=f"An error occurred during prediction: {e}")
    

if __name__=="__main__":
    app.run(host="0.0.0.0")         