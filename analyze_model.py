import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from model_trainer import FloodAwareNet

# --- CONFIG ---
MODEL_PATH = "flood_model.pth"
DATA_CSV = "dataset/data.csv"
ROOT_DIR = "dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- GRAD-CAM HELPER CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx=None):
        # Pool the gradients across channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the gradients
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU (we only care about positive influence)
        heatmap = F.relu(heatmap)
        
        # Normalize
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

def analyze_random_samples():
    # 1. Load Model
    print(f"Loading model on {DEVICE}...")
    model = FloodAwareNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to eval mode for forward pass

    # Initialize Grad-CAM on the ResNet Bottleneck (encoder4)
    # This captures the features BEFORE they are Gated/Pooled
    grad_cam = GradCAM(model, model.encoder4)

    # 2. Load Validation Data
    df = pd.read_csv(DATA_CSV)
    # Filter for "Flooded" images to see interesting Grad-CAMs
    flooded_df = df[(df['split'] == 'val') & (df['class_label'] == 1)]
    if len(flooded_df) < 5:
        sample_df = df[df['split'] == 'val'].sample(5)
    else:
        sample_df = flooded_df.sample(5)

    # Transform
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(),
        ToTensorV2()
    ])

    # 3. Visualization Loop
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    print("Generating Analysis...")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        # Load Image
        img_path = os.path.join(ROOT_DIR, row['image_path'])
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]

        # Prepare Tensor
        augmented = transform(image=original_img, mask=np.zeros((h, w)))
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        # --- FORWARD PASS ---
        # We need gradients, so we temporarily enable grad check even in eval mode
        # for the backward pass of Grad-CAM
        model.zero_grad()
        img_tensor.requires_grad = True
        
        c_logit, m_logit, features = model(img_tensor)
        
        # Predictions
        pred_prob = torch.sigmoid(c_logit).item()
        pred_mask = torch.sigmoid(m_logit).squeeze().detach().cpu().numpy()

        # --- QUANTIFICATION ---
        # Threshold mask at 0.5
        binary_mask = (pred_mask > 0.5).astype(np.float32)
        severity_index = (np.sum(binary_mask) / binary_mask.size) * 100

        # --- BACKWARD PASS (Grad-CAM) ---
        # We backpropagate from the "Flooded" neuron (index 0 for binary BCE)
        # Note: BCEWithLogits output is scalar per image, so we backprop on the logit directly.
        c_logit.backward(retain_graph=True)
        
        # Generate Heatmap
        cam_heatmap = grad_cam.generate()
        
        # --- POST-PROCESSING VISUALS ---
        # 1. Resize Grad-CAM to image size
        cam_resized = cv2.resize(cam_heatmap, (w, h))
        cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_overlay = cv2.addWeighted(original_img, 0.6, cam_color, 0.4, 0)
        
        # 2. Resize Segmentation Mask to image size
        seg_resized = cv2.resize(pred_mask, (w, h))
        seg_overlay = np.zeros_like(original_img)
        seg_overlay[:, :, 0] = np.uint8(255 * seg_resized) # Blue channel for water
        
        # 3. Final Overlay (Image + Seg)
        final_overlay = cv2.addWeighted(original_img, 0.7, seg_overlay, 0.5, 0)

        # Labels
        true_label = "Flooded" if row['class_label'] == 1 else "Dry"
        pred_label = "Flooded" if pred_prob > 0.5 else "Dry"
        
        # PLOTTING
        # Col 1: Original
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", fontsize=10)
        axes[i, 0].axis('off')

        # Col 2: Explicit Attention (Segmentation Mask)
        axes[i, 1].imshow(final_overlay)
        axes[i, 1].set_title(f"Explicit Mask (Segmentation)\nSeverity: {severity_index:.1f}%", fontsize=10, fontweight='bold', color='blue')
        axes[i, 1].axis('off')

        # Col 3: Implicit Attention (Grad-CAM)
        axes[i, 2].imshow(cam_overlay)
        axes[i, 2].set_title("Implicit Focus (Grad-CAM)\n(ResNet Bottleneck)", fontsize=10)
        axes[i, 2].axis('off')

        # Col 4: Comparison (Do they match?)
        # We show the raw heatmap vs raw mask
        axes[i, 3].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[i, 3].imshow(seg_resized, cmap='binary', alpha=0.5)
        axes[i, 3].set_title("Overlap Check\n(Heatmap vs Mask)", fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig("model_quantification_analysis.png")
    print("Saved detailed analysis to 'model_quantification_analysis.png'.")
    plt.show()


def analyze_single_image(image_path: str):
    # Resolve path (allow passing paths relative to ROOT_DIR)
    if not os.path.exists(image_path):
        candidate = os.path.join(ROOT_DIR, image_path)
        if os.path.exists(candidate):
            image_path = candidate

    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        return

    # 1. Load Model
    print(f"Loading model on {DEVICE}...")
    model = FloodAwareNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Grad-CAM hook
    grad_cam = GradCAM(model, model.encoder4)

    # Transform
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(),
        ToTensorV2()
    ])

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Failed to read image: {image_path}")
        return
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]

    augmented = transform(image=original_img, mask=np.zeros((h, w)))
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    model.zero_grad()
    img_tensor.requires_grad = True

    c_logit, m_logit, features = model(img_tensor)

    pred_prob = torch.sigmoid(c_logit).item()
    pred_mask = torch.sigmoid(m_logit).squeeze().detach().cpu().numpy()

    binary_mask = (pred_mask > 0.5).astype(np.float32)
    severity_index = (np.sum(binary_mask) / binary_mask.size) * 100

    # Backward for Grad-CAM
    c_logit.backward(retain_graph=True)
    cam_heatmap = grad_cam.generate()

    # Visuals
    cam_resized = cv2.resize(cam_heatmap, (w, h))
    cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_overlay = cv2.addWeighted(original_img, 0.6, cam_color, 0.4, 0)

    seg_resized = cv2.resize(pred_mask, (w, h))
    seg_overlay = np.zeros_like(original_img)
    seg_overlay[:, :, 0] = np.uint8(255 * seg_resized)

    final_overlay = cv2.addWeighted(original_img, 0.7, seg_overlay, 0.5, 0)

    pred_label = "Flooded" if pred_prob > 0.5 else "Dry"

    # Plot single row (1 x 4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(original_img)
    axes[0].set_title(f"Image\nPred: {pred_label} ({pred_prob:.2f})")
    axes[0].axis('off')

    axes[1].imshow(final_overlay)
    axes[1].set_title(f"Segmentation\nSeverity: {severity_index:.1f}%")
    axes[1].axis('off')

    axes[2].imshow(cam_overlay)
    axes[2].set_title("Grad-CAM")
    axes[2].axis('off')

    axes[3].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[3].imshow(seg_resized, cmap='binary', alpha=0.5)
    axes[3].set_title("Overlap")
    axes[3].axis('off')

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_name = f"single_analysis_{base}.png"
    plt.savefig(out_name)
    print(f"Saved single-image analysis to '{out_name}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model: run random samples or a single image.")
    parser.add_argument('image', nargs='?', help='Optional path to a single image to analyze')
    args = parser.parse_args()

    if args.image:
        analyze_single_image(args.image)
    else:
        analyze_random_samples()