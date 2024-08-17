from flask import Flask, request, render_template
import pandas as pd
from datetime import datetime
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Define major US holidays
US_HOLIDAYS = {
    'New Year\'s Day': '01-01',
    'Martin Luther King Jr. Day': '01-15',
    'Presidents\' Day': '02-19',
    'Memorial Day': '05-28',
    'Independence Day': '07-04',
    'Labor Day': '09-03',
    'Columbus Day': '10-08',
    'Veterans Day': '11-11',
    'Thanksgiving': '11-22',
    'Christmas Day': '12-25'
}

def get_past_sales(store, date, sales_data):
    """Calculate the average sales for the last 5 weeks for a given store."""
    # Filter data for the store
    store_data = sales_data[sales_data['Store'] == store]
    
    # Filter data for the last 5 weeks
    recent_sales = store_data[store_data['Date'] < date].sort_values(by='Date', ascending=False).head(5)
    
    # Calculate the average sales
    avg_sales = recent_sales['Weekly_Sales'].mean()
    return avg_sales

def calculate_days_to_holiday(date):
    """Calculate the days to the next major US holiday based on the current date."""
    # Convert the date to a datetime object
    current_date = datetime.strptime(date, '%Y-%m-%d')
    
    # Calculate the days to the next holiday
    min_days = float('inf')
    for holiday, holiday_date in US_HOLIDAYS.items():
        holiday_datetime = datetime.strptime(f"{current_date.year}-{holiday_date}", '%Y-%m-%d')
        days_to_holiday = (holiday_datetime - current_date).days
        
        if days_to_holiday >= 0 and days_to_holiday < min_days:
            min_days = days_to_holiday
            
    return min_days if min_days != float('inf') else np.nan

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collecting form data
        store = int(request.form.get('store'))
        date = request.form.get('date')
        sales_data = pd.read_csv('/Users/sophia/Desktop/Sales-Prediction/data/preprocessed/train_data.csv') 

        # Derived features
        past_sales = get_past_sales(store, date, sales_data)
        days_to_holiday = calculate_days_to_holiday(date)

        data = CustomData(
            store=store,
            date=date,
            past_sales=past_sales,
            days_to_holiday=days_to_holiday
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")