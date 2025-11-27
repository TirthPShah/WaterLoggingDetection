import os
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Update these paths to point to your actual folders
NON_FLOODED_DIR = "./NonFloodedAnnotations" 
FLOODED_DIR = "./FloodedAnnotations"
OUTPUT_DIR = "./dataset"

# ---------------------

def create_structure():
    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: {OUTPUT_DIR} already exists. Merging into it...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'masks', split), exist_ok=True)

def process_single_dataset(source_root, class_label, split_data):
    """
    source_root: Path to NonFloodedAnnotations or FloodedAnnotations
    class_label: 0 for NonFlooded, 1 for Flooded
    split_data: Dictionary to append records to {'train': [], 'val': []}
    """
    images_dir = os.path.join(source_root, "JPEGImages")
    masks_dir = os.path.join(source_root, "SegmentationClass") # We use Class, not Object
    
    # Get valid pairs
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.avif', '.webp'))]
    
    # Split this specific subset into train/val
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    print(f"Processing Class {class_label} from {source_root}...")
    
    for split_name, file_list in [('train', train_imgs), ('val', val_imgs)]:
        for filename in tqdm(file_list, desc=f"  -> {split_name}"):
            # Construct paths
            img_src_path = os.path.join(images_dir, filename)
            
            # Mask often has same name but .png extension
            mask_filename = os.path.splitext(filename)[0] + ".png"
            mask_src_path = os.path.join(masks_dir, mask_filename)
            
            if not os.path.exists(mask_src_path):
                # Fallback: sometimes mask has .jpg extension (rare but possible in bad exports)
                # or strictly matches image extension
                mask_src_path = os.path.join(masks_dir, filename)
                if not os.path.exists(mask_src_path):
                    print(f"    [Skipping] Mask missing for {filename}")
                    continue

            # --- PROCESS MASK ---
            # Load as Grayscale first to check for non-black pixels
            # Since background is black (0), any non-zero pixel is ROI
            mask = cv2.imread(mask_src_path)
            
            if mask is None:
                continue

            # Convert to grayscale
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Threshold: Any pixel > 0 becomes 255 (White), others 0 (Black)
            # This handles both Green and Red annotations automatically
            _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
            
            # Sanity Check: If mask is empty, warn but keep (might be fully safe road?)
            # Actually, for your task, if NonFlooded mask is empty, it means "No Road Visible"
            # which is rare. 
            if np.sum(binary_mask) == 0:
                 # Optional: Enable this if you want to skip images with no annotations
                 # print(f"    [Warning] Empty mask for {filename}")
                 pass

            # --- SAVE FILES ---
            # We rename files to avoid name collisions between the two datasets
            # e.g., flooded_img001.jpg, nonflooded_img001.jpg
            prefix = "flood" if class_label == 1 else "dry"
            new_filename = f"{prefix}_{filename}"
            new_maskname = f"{prefix}_{os.path.splitext(filename)[0]}.png"
            
            img_dst = os.path.join(OUTPUT_DIR, 'images', split_name, new_filename)
            mask_dst = os.path.join(OUTPUT_DIR, 'masks', split_name, new_maskname)
            
            shutil.copy2(img_src_path, img_dst)
            cv2.imwrite(mask_dst, binary_mask)
            
            # Add to record
            split_data.append({
                'image_path': os.path.join('images', split_name, new_filename),
                'mask_path': os.path.join('masks', split_name, new_maskname),
                'class_label': class_label,
                'split': split_name
            })

def main():
    create_structure()
    
    all_records = []
    
    # Process Non-Flooded (Label 0)
    process_single_dataset(NON_FLOODED_DIR, 0, all_records)
    
    # Process Flooded (Label 1)
    process_single_dataset(FLOODED_DIR, 1, all_records)
    
    # Save CSV
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(OUTPUT_DIR, "data.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n--- Summary ---")
    print(f"Dataset compiled at: {OUTPUT_DIR}")
    print(f"Total Images: {len(df)}")
    print(f"  - Flooded (Train+Val): {len(df[df['class_label']==1])}")
    print(f"  - Dry (Train+Val):     {len(df[df['class_label']==0])}")
    print("Check 'dataset/masks/train' to verify masks are black and white.")

if __name__ == "__main__":
    main()