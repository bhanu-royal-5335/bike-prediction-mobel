import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import helper functions from predict.py if available
try:
    from predict import load_saved_model, predict_demand
except ImportError:
    # Fallback definition of helper functions in case of import issues
    def load_saved_model(model_path='bike_sharing_model.pkl'):
        return joblib.load(model_path)
    
    def get_season_from_month(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

    def calculate_dew_point(temp, humidity):
        return temp - ((100.0 - humidity) / 5.0)

    def predict_demand(model_metadata, input_data):
        functioning_day = input_data.get('Functioning Day', 'Yes')
        if functioning_day.strip().lower() in ['no', 'n', '0', 'false']:
            return 0.0, "Functioning Day is 'No', so bike rentals are 0 by default."
        date_str = input_data.get('Date', datetime.now().strftime('%d/%m/%Y'))
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        except ValueError:
            date_obj = datetime.now()
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        day_of_week = date_obj.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        season = input_data.get('Seasons', None)
        if not season or season not in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season = get_season_from_month(month)
        seasons_encoded = model_metadata['seasons_map'].get(season, 0)
        
        holiday = input_data.get('Holiday', 'No Holiday')
        if holiday not in ['Holiday', 'No Holiday']:
            if holiday.strip().lower() in ['yes', 'holiday', '1', 'true']:
                holiday = 'Holiday'
            else:
                holiday = 'No Holiday'
        holiday_encoded = model_metadata['holiday_map'].get(holiday, 0)
        
        temp = float(input_data.get('Temperature(°C)', 20.0))
        humidity = float(input_data.get('Humidity(%)', 50.0))
        wind_speed = float(input_data.get('Wind speed (m/s)', 1.5))
        visibility = float(input_data.get('Visibility (10m)', 2000))
        
        dew_point = input_data.get('Dew point temperature(°C)', None)
        if dew_point is None or dew_point == '':
            dew_point = calculate_dew_point(temp, humidity)
        else:
            dew_point = float(dew_point)
            
        solar_radiation = float(input_data.get('Solar Radiation (MJ/m2)', 0.0))
        rainfall = float(input_data.get('Rainfall(mm)', 0.0))
        snowfall = float(input_data.get('Snowfall (cm)', 0.0))
        hour = int(input_data.get('Hour', datetime.now().hour))
        
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
        df_pred = pd.DataFrame([features_dict])[model_metadata['features']]
        prediction = model_metadata['model'].predict(df_pred)[0]
        prediction = max(0.0, prediction)
        explanation = f"Inputs: Temp={temp}°C, Humidity={humidity}%, Wind={wind_speed}m/s, Rain={rainfall}mm, Snow={snowfall}cm."
        return prediction, explanation

app = FastAPI(title="ML Model Predictor Dashboard")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models globally at startup
BIKE_MODEL_PATH = "bike_sharing_model.pkl"
WELLBEING_MODEL_PATH = "wellbeing_model.pkl"

bike_model = None
wellbeing_model = None

try:
    if os.path.exists(BIKE_MODEL_PATH):
        bike_model = load_saved_model(BIKE_MODEL_PATH)
        print("Bike model loaded successfully.")
    else:
        print("Warning: Bike sharing model not found. Run train.py first.")
except Exception as e:
    print(f"Error loading bike model: {e}")

try:
    if os.path.exists(WELLBEING_MODEL_PATH):
        wellbeing_model = joblib.load(WELLBEING_MODEL_PATH)
        print("Wellbeing model loaded successfully.")
    else:
        print("Warning: Wellbeing model not found. Run train_wellbeing.py first.")
except Exception as e:
    print(f"Error loading wellbeing model: {e}")

# Pydantic input models
class BikeInput(BaseModel):
    Date: str
    Hour: int
    Temperature: float
    Humidity: float
    WindSpeed: float
    Visibility: float
    Season: Optional[str] = None
    Holiday: str
    FunctioningDay: str
    Rainfall: float
    Snowfall: float
    SolarRadiation: float

class WellbeingInput(BaseModel):
    Entity: str
    Year: int

@app.get("/api/metadata")
def get_metadata():
    metadata = {
        "bike_model_loaded": bike_model is not None,
        "wellbeing_model_loaded": wellbeing_model is not None,
        "wellbeing_entities": [],
        "bike_metrics": {},
        "wellbeing_metrics": {}
    }
    if bike_model:
        metadata["bike_metrics"] = bike_model.get("metrics", {})
    if wellbeing_model:
        metadata["wellbeing_metrics"] = wellbeing_model.get("metrics", {})
        metadata["wellbeing_entities"] = wellbeing_model.get("unique_entities", [])
    return metadata

@app.post("/api/predict/bike")
def predict_bike(data: BikeInput):
    if not bike_model:
        raise HTTPException(status_code=503, detail="Bike sharing model is not loaded.")
    
    # Map input schema to model expectation
    inputs = {
        'Date': data.Date,
        'Hour': data.Hour,
        'Temperature(°C)': data.Temperature,
        'Humidity(%)': data.Humidity,
        'Wind speed (m/s)': data.WindSpeed,
        'Visibility (10m)': data.Visibility,
        'Seasons': data.Season,
        'Holiday': data.Holiday,
        'Functioning Day': data.FunctioningDay,
        'Rainfall(mm)': data.Rainfall,
        'Snowfall (cm)': data.Snowfall,
        'Solar Radiation (MJ/m2)': data.SolarRadiation
    }
    
    try:
        prediction, explanation = predict_demand(bike_model, inputs)
        return {
            "prediction": round(prediction, 1),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/wellbeing")
def predict_wellbeing_endpoint(data: WellbeingInput):
    if not wellbeing_model:
        raise HTTPException(status_code=503, detail="Wellbeing model is not loaded.")
    
    try:
        entity = data.Entity
        year = data.Year
        
        entity_to_id = wellbeing_model['entity_to_id']
        if entity not in entity_to_id:
            raise HTTPException(status_code=400, detail=f"Entity '{entity}' is not recognized.")
            
        entity_id = entity_to_id[entity]
        
        df_pred = pd.DataFrame([{'Entity_Encoded': entity_id, 'Year': year}])
        model = wellbeing_model['model']
        prediction = model.predict(df_pred)[0]
        prediction = max(0.0, min(100.0, prediction))
        
        return {
            "prediction": round(prediction, 4),
            "explanation": f"Forecasted Mental Disorder DALYs share for {entity} in {year} is {prediction:.4f}%."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Serve Static frontend files
@app.get("/")
def read_index():
    if not os.path.exists("static/index.html"):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
