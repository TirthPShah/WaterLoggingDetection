"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

Visualizes which regions of the image the model focuses on for predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM implementation for CNN models.
    
    Shows which parts of the image the model focuses on when making predictions.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients from (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate class activation map.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class (None for predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = (output > 0.5).float().item()
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def visualize(
        self,
        input_image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create Grad-CAM visualization overlay.
        
        Args:
            input_image: Preprocessed input tensor (1, C, H, W)
            original_image: Original image as numpy array (H, W, 3)
            target_class: Target class for CAM
            alpha: Overlay transparency (0-1)
            colormap: OpenCV colormap
            
        Returns:
            Visualization as numpy array
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to match input image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is RGB and uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap


class FloodGradCAMVisualizer:
    """High-level visualizer for flood detection models."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize visualizer.
        
        Args:
            model: Flood detection model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        
        # Find last convolutional layer
        self.target_layer = self._find_target_layer()
        
        if self.target_layer is None:
            raise ValueError("Could not find convolutional layer in model")
        
        self.gradcam = GradCAM(self.model, self.target_layer)
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the model."""
        # For custom CNN
        if hasattr(self.model, 'conv5'):
            return self.model.conv5
        elif hasattr(self.model, 'conv4'):
            return self.model.conv4
        
        # For pretrained models - find last conv layer
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        
        return target
    
    def visualize_prediction(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        denormalize_fn=None,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, float, str]:
        """
        Visualize model prediction with Grad-CAM.
        
        Args:
            image: Preprocessed image tensor (1, C, H, W)
            original_image: Original image (H, W, 3)
            denormalize_fn: Function to denormalize image
            save_path: Path to save visualization
            
        Returns:
            Tuple of (visualization, prediction_prob, prediction_class)
        """
        image = image.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            prob = output.item()
            pred_class = 'Flooded' if prob > 0.5 else 'Non-Flooded'
        
        # Generate Grad-CAM
        overlay, heatmap = self.gradcam.visualize(image, original_image)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Prediction: {pred_class}\nConfidence: {prob:.2%}',
            fontsize=12,
            fontweight='bold',
            color='green' if (pred_class == 'Flooded' and prob > 0.5) or (pred_class == 'Non-Flooded' and prob <= 0.5) else 'red'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()
        
        return overlay, prob, pred_class
    
    def visualize_batch(
        self,
        dataloader,
        num_samples: int = 5,
        save_dir: str = 'output/gradcam'
    ):
        """
        Visualize Grad-CAM for multiple samples.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        print("=" * 70)
        print(f"Generating Grad-CAM Visualizations for {num_samples} samples")
        print("=" * 70)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for images, labels in dataloader:
            for i in range(images.size(0)):
                if count >= num_samples:
                    break
                
                image = images[i:i+1]
                label = labels[i].item()
                
                # Denormalize for display
                img_denorm = self._denormalize(image[0]).permute(1, 2, 0).cpu().numpy()
                img_denorm = (img_denorm * 255).astype(np.uint8)
                
                # Visualize
                save_path = save_dir / f'gradcam_sample_{count+1}_label_{label}.png'
                self.visualize_prediction(image, img_denorm, save_path=str(save_path))
                
                count += 1
            
            if count >= num_samples:
                break
        
        print(f"\n✅ Generated {count} Grad-CAM visualizations in {save_dir}")
    
    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor using ImageNet stats."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denorm = tensor.cpu() * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        
        return denorm


if __name__ == "__main__":
    # Test Grad-CAM
    from src.flood_classifier import create_model
    
    print("Testing Grad-CAM Visualizer")
    
    # Create model
    model = create_model('custom')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 244, 244)
    dummy_image = (np.random.rand(244, 244, 3) * 255).astype(np.uint8)
    
    # Test visualizer
    visualizer = FloodGradCAMVisualizer(model, device)
    print(f"✅ Target layer found: {visualizer.target_layer}")
    
    print("\n✅ Grad-CAM test completed!")
