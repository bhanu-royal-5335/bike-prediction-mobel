import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(filepath):
    print(f"Loading wellbeing dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    print("Preprocessing wellbeing dataset...")
    
    # Identify target and features
    target_col = 'DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col]).copy()
    
    # We will use Entity (Country) and Year as features
    # Create entity mapping
    unique_entities = sorted(df['Entity'].unique())
    entity_to_id = {entity: idx for idx, entity in enumerate(unique_entities)}
    
    df['Entity_Encoded'] = df['Entity'].map(entity_to_id)
    
    # Also map Code (in case there are some mismatches, but Entity is sufficient)
    unique_codes = sorted(df['Code'].dropna().unique())
    code_to_id = {code: idx for idx, code in enumerate(unique_codes)}
    
    # Save the mapping of entity name to entity ID so we can reuse it in prediction
    return df, entity_to_id, target_col

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
    print(f"Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    
    return {
        'model_name': model_name,
        'model': model,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred
    }

def main():
    dataset_file = '1776165773-P1-Cognitive Wellbeing Monitoring.csv'
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Wellbeing dataset file '{dataset_file}' not found.")
        
    df = load_data(dataset_file)
    processed_df, entity_to_id, target_col = preprocess_data(df)
    
    # Features are 'Entity_Encoded' and 'Year'
    X = processed_df[['Entity_Encoded', 'Year']]
    y = processed_df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }
    
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        eval_res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(eval_res)
        
    # Select Best Model based on Test R2 Score
    best_res = max(results, key=lambda x: x['test_r2'])
    print(f"\nBest Model: {best_res['model_name']} with Test R2 = {best_res['test_r2']:.4f}")
    
    # Save the Best Model
    model_save_path = 'wellbeing_model.pkl'
    metadata = {
        'model': best_res['model'],
        'features': ['Entity_Encoded', 'Year'],
        'entity_to_id': entity_to_id,
        'unique_entities': sorted(list(entity_to_id.keys())),
        'metrics': {
            'model_name': best_res['model_name'],
            'test_r2': float(best_res['test_r2']),
            'test_mae': float(best_res['test_mae']),
            'test_rmse': float(best_res['test_rmse'])
        }
    }
    joblib.dump(metadata, model_save_path)
    print(f"Wellbeing model saved successfully to {model_save_path}")

if __name__ == '__main__':
    main()
