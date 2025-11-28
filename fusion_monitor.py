import cv2
import torch
import numpy as np
import time
import csv
import os
import requests
import joblib
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from model_trainer import FloodAwareNet  # Must be in the same folder

# --- CONFIGURATION ---
# RTSP Settings (Using the Empty Password format you confirmed)
RTSP_URL = "rtsp://admin:@59.92.102.53:554/live"

# Location (Ahmedabad)
LAT = 23.0225
LON = 72.5714

# Model Paths
VISION_MODEL_PATH = "flood_model.pth"
WEATHER_MODEL_PATH = "weather_xgb.model"
CSV_LOG_FILE = "fusion_system_log.csv"

# Settings
SAMPLING_INTERVAL = 30  # Seconds
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- PREPROCESSING ---
transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(),
    ToTensorV2()
])

# --- MODULE 1: WEATHER FORECASTING ---
def get_live_weather_features(lat, lon):
    """
    Fetches real-time weather and formats it EXACTLY how the XGBoost model expects it.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m",
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json().get('current', {})
        
        # Create DataFrame with exact column names used in training
        features = pd.DataFrame([{
            'temperature_2m': data.get('temperature_2m', 0),
            'relative_humidity_2m': data.get('relative_humidity_2m', 0),
            'surface_pressure': data.get('surface_pressure', 0),
            'cloud_cover': data.get('cloud_cover', 0),
            'wind_speed_10m': data.get('wind_speed_10m', 0)
        }])
        return features
    except Exception as e:
        print(f" [!] Weather API Error: {e}")
        return None

def load_weather_model():
    print("Loading XGBoost Weather Model...")
    try:
        return joblib.load(WEATHER_MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: {WEATHER_MODEL_PATH} not found. Run train_weather_model.py first!")
        exit()

# --- MODULE 2: VISION INFERENCE ---
def load_vision_model():
    print(f"Loading Attention-Gated Vision Model on {DEVICE}...")
    # Initialize with the same bg_gain=0.05 used in fine-tuning
    model = FloodAwareNet(bg_gain=0.05).to(DEVICE)
    try:
        model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
    except:
        print("Warning: Strict loading failed. Attempting partial load...")
        model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    return model

def get_fresh_frame(url):
    """Connects, grabs a frame, and disconnects to avoid buffer lag."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened(): return None
    
    # Flush buffer by reading a few frames
    for _ in range(5): 
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def vision_inference(model, frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Dummy mask
    augmented = transform(image=rgb, mask=np.zeros((h, w), dtype=np.uint8))
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        c_logit, m_logit, _ = model(img_tensor)
        flood_prob = torch.sigmoid(c_logit).item()
        pred_mask = torch.sigmoid(m_logit).squeeze().cpu().numpy()
        
    # Calculate Severity (Percentage of image covered by water)
    binary_mask = (pred_mask > 0.5).astype(np.float32)
    severity = (np.sum(binary_mask) / binary_mask.size) * 100
    return flood_prob, severity, pred_mask

# --- MODULE 3: FUSION LOGIC ---
def assess_risk(flood_severity, rain_prob):
    """
    Combines 'Visual Severity' (Current) and 'Rain Probability' (Future)
    """
    if rain_prob > 60.0:
        return "HIGH", "Flooding + Rain Imminent"
    elif rain_prob > 85.0:
        return "WARNING", "Dry but Storm Incoming"
    else:
        return "LOW", "Normal Conditions"

# --- MAIN LOOP ---
def main():
    print("\n=== INITIALIZING FUSION SYSTEM ===")
    vision_model = load_vision_model()
    weather_model = load_weather_model()
    
    # Init CSV
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Visual_Severity", "Rain_Prob_1h", "Risk_Level", "Status_Message"])

    print(f"\nSystem Online. Monitoring {RTSP_URL}")
    print(f"Weather Location: {LAT}, {LON}")
    print(f"Sampling Rate: {SAMPLING_INTERVAL} seconds\n")
    
    is_first_run = True

    try:
        while True:
            cycle_start = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Sampling...")

            # 1. Get Weather Prediction
            rain_prob = 0.0
            weather_feats = get_live_weather_features(LAT, LON)
            if weather_feats is not None:
                # Predict probability of class 1 (Rain)
                rain_prob = weather_model.predict_proba(weather_feats)[0][1] * 100

            # 2. Get Vision Status
            frame = get_fresh_frame(RTSP_URL)
            if frame is not None:
                flood_prob, severity, mask = vision_inference(vision_model, frame)
                
                # 3. Fuse & Decide
                risk_level, message = assess_risk(severity, rain_prob)
                
                # Console Report
                print(f" > Vision Severity: {severity:.1f}%")
                print(f" > Rain Forecast:   {rain_prob:.1f}%")
                print(f" > SYSTEM STATUS:   {risk_level} ({message})")

                # 4. First Run Visual Check
                if is_first_run:
                    h, w = frame.shape[:2]
                    # Resize mask for display
                    mask_vis = cv2.applyColorMap(np.uint8(255 * cv2.resize(mask, (w, h))), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.7, mask_vis, 0.3, 0)
                    
                    # Dashboard UI
                    cv2.rectangle(overlay, (0, 0), (w, 120), (0,0,0), -1)
                    cv2.putText(overlay, f"RISK: {risk_level}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(overlay, f"Severity: {severity:.1f}% | Rain Prob: {rain_prob:.0f}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Fusion System Check", overlay)
                    print("\n>>> Check the popup window. Press any key to continue background monitoring. <<<")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    is_first_run = False

                # 5. Log Data
                with open(CSV_LOG_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([timestamp, f"{severity:.2f}", f"{rain_prob:.2f}", risk_level, message])

            else:
                print(" > Error: Camera Offline/Unreachable")

            # Wait for next cycle
            elapsed = time.time() - cycle_start
            time.sleep(max(0, SAMPLING_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\nSystem Shutdown.")

if __name__ == "__main__":
    main()
