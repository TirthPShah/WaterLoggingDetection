import cv2
import torch
import numpy as np
import time
import csv
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from model_trainer import FloodAwareNet # Imports your model class
from dotenv import load_dotenv

# --- CONFIGURATION ---
# The URL you found successful in VLC
RTSP_URL = "rtsp://admin:@59.92.102.53:554/live" 
# Note the colon after admin (admin:) handles the empty password field

MODEL_PATH = "flood_model.pth"
CSV_LOG_FILE = "live_flood_data.csv"
SAMPLING_INTERVAL = 30 # Seconds
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- PREPROCESSING ---
transform = A.Compose([
    A.Resize(320, 320), # Must match your training size
    A.Normalize(),
    ToTensorV2()
])

def load_model():
    print(f"Loading Fine-Tuned Flood Model on {DEVICE}...")
    # Initialize with the same bg_gain you trained with
    model = FloodAwareNet(bg_gain=0.05).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Warning: Strict loading failed, trying non-strict. Error: {e}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    
    model.eval()
    return model

def get_fresh_frame(url):
    """
    Connects, grabs a single fresh frame, and disconnects.
    This avoids buffer lag issues common with long sleep times.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    
    # Read a few frames to flush initialization artifacts
    for _ in range(5):
        ret, frame = cap.read()
    
    # Grab the actual frame we want
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    return frame

def process_frame(model, frame):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Dummy mask for Albumentations
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    
    augmented = transform(image=rgb_frame, mask=dummy_mask)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        c_logit, m_logit, _ = model(img_tensor)
        
        # Predictions
        flood_prob = torch.sigmoid(c_logit).item()
        pred_mask = torch.sigmoid(m_logit).squeeze().cpu().numpy()
        
    # Severity Calculation (Pixels > 50% confidence)
    binary_mask = (pred_mask > 0.5).astype(np.float32)
    severity_index = (np.sum(binary_mask) / binary_mask.size) * 100
    
    return flood_prob, severity_index, pred_mask

def main():
    model = load_model()
    
    # Init CSV
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Flood_Probability", "Severity_Index"])

    print(f"Starting Monitoring on {RTSP_URL}")
    print(f"Sampling every {SAMPLING_INTERVAL} seconds...")

    is_first_run = True # Flag to trigger visualization only once

    try:
        while True:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] Connecting to Camera...")
            frame = get_fresh_frame(RTSP_URL)
            
            if frame is None:
                print(f"[{timestamp}] Failed to connect/read. Retrying in 30s.")
            else:
                # Inference
                prob, severity, mask = process_frame(model, frame)

                # --- FIRST RUN VISUALIZATION ---
                if is_first_run:
                    print(">>> FIRST RUN: Generating visualization. Check the popup window. <<<")
                    h, w = frame.shape[:2]
                    # Resize mask to original frame size
                    mask_resized = cv2.resize(mask, (w, h))
                    
                    # Create visualizations
                    # 1. Mask view (grayscale converted to BGR for stacking)
                    mask_vis = cv2.cvtColor(np.uint8(255 * mask_resized), cv2.COLOR_GRAY2BGR)
                    
                    # 2. Overlay view
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
                    
                    # Combine horizontally: Original | Mask | Overlay
                    combined_vis = np.hstack((frame, mask_vis, overlay))
                    
                    # Add status text
                    status_text = f"Prediction: {'FLOODED' if prob > 0.5 else 'DRY'} (Conf: {prob:.2f}, Severity: {severity:.1f}%)"
                    cv2.putText(combined_vis, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(combined_vis, "Original Frame", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(combined_vis, "Segmentation Mask", (w+20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(combined_vis, "Model Overlay", (w*2+20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # Show and wait for user input
                    cv2.imshow("First Run System Check (Press any key to start monitoring)", combined_vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(">>> Visualization closed. Beginning background monitoring... <<<")
                    is_first_run = False
                # -------------------------------
                
                # Console Output
                status = "FLOODED" if prob > 0.5 else "DRY"
                print(f"[{timestamp}] Status: {status} | Conf: {prob:.2f} | Severity: {severity:.2f}%")
                
                # CSV Logging
                with open(CSV_LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, f"{prob:.4f}", f"{severity:.4f}"])
                
                # Optional: Save Image if Flooded (for debugging)
                if prob > 0.5 and not is_first_run: # Don't double save on first run
                    filename = f"detected_flood_{timestamp.replace(':', '-')}.jpg"
                    # Overlay re-calculation for saving
                    mask_resized_save = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    heatmap_save = cv2.applyColorMap(np.uint8(255 * mask_resized_save), cv2.COLORMAP_JET)
                    overlay_save = cv2.addWeighted(frame, 0.7, heatmap_save, 0.3, 0)
                    cv2.imwrite(filename, overlay_save)
                    print(f"   -> Alert! Image saved to {filename}")

            # Sleep precisely to maintain interval
            elapsed = time.time() - start_time
            sleep_time = max(0, SAMPLING_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nMonitoring Stopped.")

if __name__ == "__main__":
    main()