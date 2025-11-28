import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- CONFIG ---
# Coordinates for Ahmedabad (Match your RTSP location)
LAT = 11.3655
LON = 76.7843
MODEL_FILE = "weather_xgb.model"

def fetch_historical_data():
    print("Fetching 90 days of historical weather data...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": (pd.Timestamp.now() - pd.Timedelta(days=92)).strftime("%Y-%m-%d"),
        "end_date": (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m,precipitation,rain",
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    
    print(f"Fetched {len(df)} hourly records.")
    return df

def prepare_dataset(df):
    """
    Goal: Predict if it will rain in the NEXT hour based on current metrics.
    Target: 1 if rain > 0.1mm in the next hour, else 0.
    """
    # Features (Current State)
    feature_cols = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'cloud_cover', 'wind_speed_10m']
    
    X = df[feature_cols].copy()
    
    # Target (Future State)
    # Shift 'rain' column up by 1 row (1 hour look-ahead)
    # If using 30-min data, logic is same. Open-Meteo Archive is hourly.
    # We define "Rain Event" as > 0.1mm precipitation
    df['future_rain'] = df['rain'].shift(-1)
    y = (df['future_rain'] > 0.1).astype(int)
    
    # Remove last row (NaN target)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    return X, y

def train_and_save():
    df = fetch_historical_data()
    X, y = prepare_dataset(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train XGBoost
    print(f"Training XGBoost on {len(X_train)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds))
    
    # Save
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save()