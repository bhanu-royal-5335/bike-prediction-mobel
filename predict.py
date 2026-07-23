import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

def get_season_from_month(month):
    # Standard meteorological seasons
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def calculate_dew_point(temp, humidity):
    # Simple rule of thumb approximation: Td ≈ T - ((100 - RH)/5)
    return temp - ((100.0 - humidity) / 5.0)

def load_saved_model(model_path='bike_sharing_model.pkl'):
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found.")
        print("Please run 'train.py' first to train and save the model.")
        sys.exit(1)
    
    print(f"Loading trained model metadata from '{model_path}'...")
    return joblib.load(model_path)

def predict_demand(model_metadata, input_data):
    """
    input_data: dict containing keys like Date, Hour, Temperature, Humidity, etc.
    """
    # 1. Check Functioning Day first (Domain knowledge rule)
    functioning_day = input_data.get('Functioning Day', 'Yes')
    if functioning_day.strip().lower() in ['no', 'n', '0', 'false']:
        return 0.0, "Functioning Day is 'No', so bike rentals are 0 by default."
        
    # 2. Extract and format date features
    date_str = input_data.get('Date', datetime.now().strftime('%d/%m/%Y'))
    try:
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
    except ValueError:
        print(f"Error parsing date '{date_str}'. Expected format is dd/mm/yyyy. Using current date.")
        date_obj = datetime.now()
        
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    day_of_week = date_obj.weekday()  # Monday=0, Sunday=6
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 3. Handle Seasons (auto-detect if not provided or invalid)
    season = input_data.get('Seasons', None)
    if not season or season not in ['Winter', 'Spring', 'Summer', 'Autumn']:
        season = get_season_from_month(month)
        
    seasons_encoded = model_metadata['seasons_map'].get(season, 0)
    
    # 4. Handle Holiday (default to No Holiday)
    holiday = input_data.get('Holiday', 'No Holiday')
    if holiday not in ['Holiday', 'No Holiday']:
        if holiday.strip().lower() in ['yes', 'holiday', '1', 'true']:
            holiday = 'Holiday'
        else:
            holiday = 'No Holiday'
    holiday_encoded = model_metadata['holiday_map'].get(holiday, 0)
    
    # 5. Handle weather numerical parameters
    temp = float(input_data.get('Temperature(°C)', 20.0))
    humidity = float(input_data.get('Humidity(%)', 50.0))
    wind_speed = float(input_data.get('Wind speed (m/s)', 1.5))
    visibility = float(input_data.get('Visibility (10m)', 2000))
    
    # If dew point is not provided, approximate it
    dew_point = input_data.get('Dew point temperature(°C)', None)
    if dew_point is None or dew_point == '':
        dew_point = calculate_dew_point(temp, humidity)
    else:
        dew_point = float(dew_point)
        
    solar_radiation = float(input_data.get('Solar Radiation (MJ/m2)', 0.0))
    rainfall = float(input_data.get('Rainfall(mm)', 0.0))
    snowfall = float(input_data.get('Snowfall (cm)', 0.0))
    hour = int(input_data.get('Hour', datetime.now().hour))
    
    # 6. Build input dataframe matching exact training columns
    # Order of training columns:
    # ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 
    #  'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 
    #  'Seasons', 'Holiday', 'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend']
    
    features_dict = {
        'Hour': hour,
        'Temperature(°C)': temp,
        'Humidity(%)': humidity,
        'Wind speed (m/s)': wind_speed,
        'Visibility (10m)': visibility,
        'Dew point temperature(°C)': dew_point,
        'Solar Radiation (MJ/m2)': solar_radiation,
        'Rainfall(mm)': rainfall,
        'Snowfall (cm)': snowfall,
        'Seasons': seasons_encoded,
        'Holiday': holiday_encoded,
        'Year': year,
        'Month': month,
        'Day': day,
        'DayOfWeek': day_of_week,
        'IsWeekend': is_weekend
    }
    
    # Create DataFrame in the exact sequence expected by model
    features_ordered = model_metadata['features']
    df_pred = pd.DataFrame([features_dict])[features_ordered]
    
    # 7. Predict
    model = model_metadata['model']
    prediction = model.predict(df_pred)[0]
    
    # Since bike rentals cannot be negative, clip to 0
    prediction = max(0.0, prediction)
    
    explanation = (
        f"Input details: Date={date_obj.strftime('%Y-%m-%d')} ({season}), Hour={hour:02d}:00, "
        f"Temp={temp}°C, Humidity={humidity}%, Wind={wind_speed}m/s, DewPoint={dew_point:.1f}°C, "
        f"Rain={rainfall}mm, Snow={snowfall}cm, Holiday={holiday}."
    )
    
    return prediction, explanation

def run_interactive(model_metadata):
    print("\n==================================================")
    print("      Bike Sharing Demand Predictor - Interactive  ")
    print("==================================================")
    print("Press Enter to use default values shown in [brackets]\n")
    
    # Get interactive inputs
    date_input = input(f"Date (dd/mm/yyyy) [{datetime.now().strftime('%d/%m/%Y')}]: ").strip()
    if not date_input:
        date_input = datetime.now().strftime('%d/%m/%Y')
        
    hour_input = input(f"Hour (0-23) [{datetime.now().hour}]: ").strip()
    if not hour_input:
        hour_input = datetime.now().hour
    else:
        hour_input = int(hour_input)
        
    temp_input = input("Temperature (°C) [20.0]: ").strip()
    temp_input = float(temp_input) if temp_input else 20.0
    
    humidity_input = input("Humidity (%) [50.0]: ").strip()
    humidity_input = float(humidity_input) if humidity_input else 50.0
    
    wind_input = input("Wind speed (m/s) [1.5]: ").strip()
    wind_input = float(wind_input) if wind_input else 1.5
    
    visibility_input = input("Visibility (10m) [2000]: ").strip()
    visibility_input = float(visibility_input) if visibility_input else 2000.0
    
    # Seasons: Auto-detect from date if blank
    month = datetime.strptime(date_input, '%d/%m/%Y').month
    default_season = get_season_from_month(month)
    season_input = input(f"Season (Winter/Spring/Summer/Autumn) [{default_season}]: ").strip()
    if not season_input:
        season_input = default_season
        
    holiday_input = input("Is it a Holiday? (Holiday/No Holiday) [No Holiday]: ").strip()
    if not holiday_input:
        holiday_input = "No Holiday"
        
    functioning_input = input("Is the bike sharing service functioning? (Yes/No) [Yes]: ").strip()
    if not functioning_input:
        functioning_input = "Yes"
        
    rain_input = input("Rainfall (mm) [0.0]: ").strip()
    rain_input = float(rain_input) if rain_input else 0.0
    
    snow_input = input("Snowfall (cm) [0.0]: ").strip()
    snow_input = float(snow_input) if snow_input else 0.0
    
    solar_input = input("Solar Radiation (MJ/m2) [0.0]: ").strip()
    solar_input = float(solar_input) if solar_input else 0.0

    # Build input dictionary
    inputs = {
        'Date': date_input,
        'Hour': hour_input,
        'Temperature(°C)': temp_input,
        'Humidity(%)': humidity_input,
        'Wind speed (m/s)': wind_input,
        'Visibility (10m)': visibility_input,
        'Seasons': season_input,
        'Holiday': holiday_input,
        'Functioning Day': functioning_input,
        'Rainfall(mm)': rain_input,
        'Snowfall (cm)': snow_input,
        'Solar Radiation (MJ/m2)': solar_input,
        'Dew point temperature(°C)': None # Will be calculated
    }
    
    pred, explanation = predict_demand(model_metadata, inputs)
    print("\n------------------- PREDICTION -------------------")
    print(f"Forecasted Bike Rental Demand: {round(pred)} bikes")
    print(f"Details: {explanation}")
    print("==================================================")

def main():
    parser = argparse.ArgumentParser(description="Predict Bike Sharing Demand")
    parser.add_argument('--date', type=str, help="Date in dd/mm/yyyy format")
    parser.add_argument('--hour', type=int, help="Hour of the day (0-23)")
    parser.add_argument('--temp', type=float, help="Temperature in °C")
    parser.add_argument('--humidity', type=float, help="Humidity percentage")
    parser.add_argument('--wind', type=float, help="Wind speed (m/s)")
    parser.add_argument('--visibility', type=float, help="Visibility (10m)")
    parser.add_argument('--season', type=str, choices=['Winter', 'Spring', 'Summer', 'Autumn'], help="Season name")
    parser.add_argument('--holiday', type=str, choices=['Holiday', 'No Holiday'], help="Holiday status")
    parser.add_argument('--functioning', type=str, choices=['Yes', 'No'], default='Yes', help="Is it a functioning day?")
    parser.add_argument('--rain', type=float, default=0.0, help="Rainfall (mm)")
    parser.add_argument('--snow', type=float, default=0.0, help="Snowfall (cm)")
    parser.add_argument('--solar', type=float, default=0.0, help="Solar Radiation (MJ/m2)")
    parser.add_argument('--model', type=str, default='bike_sharing_model.pkl', help="Path to the saved model file")
    
    args = parser.parse_args()
    
    model_metadata = load_saved_model(args.model)
    
    # Check if command-line arguments are provided, otherwise run interactive mode
    if len(sys.argv) > 1 and any(arg != '--model' and not arg.startswith('bike_sharing_model') for arg in sys.argv[1:]):
        # Format arguments into dictionary
        inputs = {
            'Date': args.date,
            'Hour': args.hour,
            'Temperature(°C)': args.temp,
            'Humidity(%)': args.humidity,
            'Wind speed (m/s)': args.wind,
            'Visibility (10m)': args.visibility,
            'Seasons': args.season,
            'Holiday': args.holiday,
            'Functioning Day': args.functioning,
            'Rainfall(mm)': args.rain,
            'Snowfall (cm)': args.snow,
            'Solar Radiation (MJ/m2)': args.solar
        }
        # Filter None values to let defaults apply
        inputs = {k: v for k, v in inputs.items() if v is not None}
        
        pred, explanation = predict_demand(model_metadata, inputs)
        print(f"\nPredicted Bike Rental Demand: {round(pred)} bikes")
        print(f"Details: {explanation}")
    else:
        run_interactive(model_metadata)

if __name__ == '__main__':
    main()
