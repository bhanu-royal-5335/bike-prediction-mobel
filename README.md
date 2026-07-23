# Bike Sharing & Cognitive Wellbeing Prediction Dashboard

An interactive, dual-purpose Machine Learning application featuring a FastAPI backend and a responsive web dashboard. The application provides two main predictive functionalities:
1. **Bike Sharing Demand Prediction**: Forecasts hourly bike rental demand based on weather, seasonality, and calendar factors.
2. **Cognitive Wellbeing Monitoring**: Projects mental disorder DALYs (Disability-Adjusted Life Years) share by entity/country and year.

---

## 🚀 Features

- **Double ML Pipelines**: Linear Regression, Random Forest, and Gradient Boosting models trained and evaluated for both datasets, automatically saving the best-performing model.
- **FastAPI Web Service**: A high-performance asynchronous API backend serving predictions and project metadata.
- **Interactive Web Dashboard**: Sleek frontend (HTML5/CSS3/JavaScript) served directly from the FastAPI application.
- **Interactive CLI tool**: Run predictions directly from the command line in either manual or interactive console mode.
- **Performance Evaluation Plots**: Generates actual vs. predicted and feature importance plots during bike demand model training.

---

## 📁 Project Directory Structure

```text
├── 1776165773-P1-Cognitive Wellbeing Monitoring.csv    # Wellbeing Dataset
├── 1776241192-P4-Bike Sharing Demand Prediction.csv    # Bike Sharing Dataset
├── app.py                                              # FastAPI Web Server Application
├── train.py                                            # Model Training script for Bike Sharing
├── train_wellbeing.py                                  # Model Training script for Wellbeing
├── predict.py                                          # CLI prediction tool & helper utilities
├── bike_sharing_model.pkl                              # Trained Bike Sharing Model (Metadata & weights)
├── wellbeing_model.pkl                                 # Trained Wellbeing Model (Metadata & weights)
├── requirements.txt                                    # Python dependencies
├── static/                                             # Frontend static web files
│   ├── index.html                                      # Dashboard HTML structure
│   ├── style.css                                       # Dashboard styling
│   └── script.js                                       # Dashboard interactive logic
└── plots/                                              # Visualizations generated during training
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Create a Virtual Environment (Recommended)
Set up a clean virtual environment to manage dependencies:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

To train the machine learning models on the raw datasets, execute the training scripts. The scripts will:
- Load and preprocess the datasets.
- Train multiple regression models (Linear Regression, Random Forest, Gradient Boosting).
- Select and serialize the best-performing model to a `.pkl` file.
- (For Bike Sharing) Generate training evaluation plots in the `plots/` directory.

```bash
# Train Bike Sharing Demand Predictor
python train.py

# Train Cognitive Wellbeing Projector
python train_wellbeing.py
```

---

## 🖥️ Usage

### 1. Launching the Web Dashboard
Run the FastAPI web application to serve the dashboard locally:
```bash
python app.py
```
By default, the server will start at **`http://localhost:8000`**. Open this URL in your web browser to access the dashboard.

### 2. Running Command Line Predictions
You can use `predict.py` to make bike sharing predictions from the terminal:

#### Interactive Console Mode:
```bash
python predict.py
```
This will guide you step-by-step through entering parameters like Temperature, Humidity, Season, etc.

#### Direct Command Line Input:
```bash
python predict.py --temp 25.5 --humidity 45 --hour 14 --season Summer --holiday "No Holiday"
```

---

## 🔌 API Documentation

FastAPI automatically generates interactive Swagger API documentation. You can view it by navigating to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints
- **`GET /api/metadata`**: Returns model statuses, feature lists, and performance metrics (R², MAE, RMSE).
- **`POST /api/predict/bike`**: Predicts hourly bike rental counts.
- **`POST /api/predict/wellbeing`**: Predicts the cognitive wellbeing DALYs percentage.
