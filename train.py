import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(filepath):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_and_feature_engineering(df):
    print("Performing preprocessing and feature engineering...")
    
    # 1. Parse Date column (format dd/mm/yyyy)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop original Date column
    df = df.drop(columns=['Date'])
    
    # 2. Filter dataset for Functioning Day == 'Yes'
    # For non-functioning days, rented bike count is 0. We'll handle this in the prediction script.
    # Therefore, we train the model only on data where bikes were actually available for rent.
    df_train = df[df['Functioning Day'] == 'Yes'].copy()
    df_train = df_train.drop(columns=['Functioning Day'])
    
    # 3. Categorical encoding
    # Map Seasons: Winter=0, Spring=1, Summer=2, Autumn=3
    seasons_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    df_train['Seasons'] = df_train['Seasons'].map(seasons_map)
    
    # Map Holiday: Holiday=1, No Holiday=0
    holiday_map = {'Holiday': 1, 'No Holiday': 0}
    df_train['Holiday'] = df_train['Holiday'].map(holiday_map)
    
    # Verify no missing values in encoded columns
    if df_train['Seasons'].isnull().any() or df_train['Holiday'].isnull().any():
        print("Warning: Unrecognized values in Seasons or Holiday columns. Filling with defaults.")
        df_train['Seasons'] = df_train['Seasons'].fillna(0)
        df_train['Holiday'] = df_train['Holiday'].fillna(0)
        
    return df_train

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
    print(f"Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
    print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
    
    return {
        'model_name': model_name,
        'model': model,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred
    }

def save_plots(best_model_results, X_test, y_test, features, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=best_model_results['y_test_pred'], alpha=0.3, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted Bike Rentals ({best_model_results['model_name']})")
    plt.xlabel("Actual Rented Bike Count")
    plt.ylabel("Predicted Rented Bike Count")
    plt.tight_layout()
    plot_path_avp = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path_avp)
    plt.close()
    print(f"Saved Actual vs Predicted plot to {plot_path_avp}")
    
    # 2. Feature Importance Plot (if model supports it)
    model = best_model_results['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
        plt.title(f"Feature Importances ({best_model_results['model_name']})")
        plt.xlabel("Relative Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plot_path_fi = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path_fi)
        plt.close()
        print(f"Saved Feature Importance plot to {plot_path_fi}")

def main():
    # File paths
    dataset_file = '1776241192-P4-Bike Sharing Demand Prediction.csv'
    
    if not os.path.exists(dataset_file):
        # Fallback to absolute path or check if we are in the wrong directory
        raise FileNotFoundError(f"Dataset file '{dataset_file}' not found. Please make sure to run the script from the directory containing the dataset.")
        
    df = load_data(dataset_file)
    processed_df = preprocess_and_feature_engineering(df)
    
    # Define Target and Features
    target_col = 'Rented Bike Count'
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    features = list(X.columns)
    
    print("\nFeatures used for training:")
    print(features)
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Initialize Models
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        eval_res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(eval_res)
        
    # Select Best Model based on Test R2 Score
    best_res = max(results, key=lambda x: x['test_r2'])
    print(f"\nBest Model: {best_res['model_name']} with Test R2 = {best_res['test_r2']:.4f}")
    
    # Save the Best Model
    model_save_path = 'bike_sharing_model.pkl'
    metadata = {
        'model': best_res['model'],
        'features': features,
        'seasons_map': {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3},
        'holiday_map': {'Holiday': 1, 'No Holiday': 0},
        'metrics': {
            'model_name': best_res['model_name'],
            'test_r2': float(best_res['test_r2']),
            'test_mae': float(best_res['test_mae']),
            'test_rmse': float(best_res['test_rmse'])
        }
    }
    joblib.dump(metadata, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    # Save visualization plots
    save_plots(best_res, X_test, y_test, features)
    
    print("\nTraining workflow completed successfully!")

if __name__ == '__main__':
    main()
