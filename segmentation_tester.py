import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_trainer import FloodAwareNet  # Import your model class

# --- CONFIG ---
MODEL_PATH = "flood_model.pth"
DATA_CSV = "dataset/data.csv"
ROOT_DIR = "dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def visualize():
    # 1. Load Model
    model = FloodAwareNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Validation Data
    df = pd.read_csv(DATA_CSV)
    
    # Check if we have enough validation images, else take all
    n_samples = min(5, len(df[df['split'] == 'val']))
    val_df = df[df['split'] == 'val'].sample(n_samples)

    # Transform
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(),
        ToTensorV2()
    ])

    # 3. Plot
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    print(f"Visualizing {n_samples} random results on {DEVICE}...")

    for i, (_, row) in enumerate(val_df.iterrows()):
        # Load Image
        img_path = os.path.join(ROOT_DIR, row['image_path'])
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Could not load {img_path}")
            continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # --- THE FIX IS HERE ---
        # Create a dummy mask matching the ORIGINAL image size
        h, w = original_img.shape[:2]
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Apply Transform
        augmented = transform(image=original_img, mask=dummy_mask) 
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        # -----------------------

        # Inference
        with torch.no_grad():
            c_logit, m_logit, _ = model(img_tensor)
            pred_prob = torch.sigmoid(c_logit).item()
            pred_mask = torch.sigmoid(m_logit).squeeze().cpu().numpy()

        # Prepare for display
        # Resize predicted mask back to original image size for display overlay
        pred_mask_resized = cv2.resize(pred_mask, (w, h))
        
        # Create Heatmap Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * pred_mask_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Title Logic
        true_label = "Flooded" if row['class_label'] == 1 else "Dry"
        pred_label = "Flooded" if pred_prob > 0.5 else "Dry"
        color = "green" if true_label == pred_label else "red"

        # Handle axes if n_samples=1 (matplotlib quirk)
        ax_row = axes if n_samples == 1 else axes[i]

        # Plot 1: Original
        ax_row[0].imshow(original_img)
        ax_row[0].set_title(f"True: {true_label}", fontsize=10)
        ax_row[0].axis('off')

        # Plot 2: Attention Mask (What the model is looking at)
        ax_row[1].imshow(pred_mask_resized, cmap='gray')
        ax_row[1].set_title("Learned Attention Mask", fontsize=10)
        ax_row[1].axis('off')

        # Plot 3: Overlay + Prediction
        ax_row[2].imshow(overlay)
        ax_row[2].set_title(f"Pred: {pred_label} ({pred_prob:.2f})", color=color, fontsize=10, fontweight='bold')
        ax_row[2].axis('off')

    plt.tight_layout()
    plt.savefig("results_visualization.png")
    print("Saved visualization to 'results_visualization.png'. Open it to verify!")
    plt.show()

if __name__ == "__main__":
    visualize()