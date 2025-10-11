"""
Waterlogging detection model using semantic segmentation
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Tuple, Optional
import numpy as np


class WaterloggingDetector:
    """Waterlogging detection model wrapper"""
    
    def __init__(
        self,
        model_name: str = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 1,
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Initialize waterlogging detector
        
        Args:
            model_name: Segmentation architecture (unet, deeplabv3, fpn, etc.)
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            num_classes: Number of output classes (1 for binary segmentation)
            device: Device to run model on
            threshold: Probability threshold for waterlogging detection
        """
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _create_model(self) -> nn.Module:
        """Create segmentation model"""
        if self.model_name.lower() == "unet":
            model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None  # We'll apply sigmoid manually
            )
        elif self.model_name.lower() == "deeplabv3":
            model = smp.DeepLabV3(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == "deeplabv3+":
            model = smp.DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == "fpn":
            model = smp.FPN(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == "pspnet":
            model = smp.PSPNet(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def predict(
        self,
        image_tensor: torch.Tensor,
        return_probability: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict waterlogging mask for input image
        
        Args:
            image_tensor: Input tensor (C, H, W) or (B, C, H, W)
            return_probability: Return probability map instead of binary mask
            
        Returns:
            Tuple of (mask, metadata)
            - mask: Binary mask or probability map (H, W)
            - metadata: Dict with prediction statistics
        """
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        prob_map = probabilities.squeeze().cpu().numpy()
        
        # Calculate metadata
        metadata = {
            'max_probability': float(prob_map.max()),
            'mean_probability': float(prob_map.mean()),
            'waterlogged_area_ratio': float((prob_map > self.threshold).sum() / prob_map.size),
            'threshold': self.threshold
        }
        
        if return_probability:
            return prob_map, metadata
        else:
            # Apply threshold
            binary_mask = (prob_map > self.threshold).astype(np.uint8)
            return binary_mask, metadata
    
    def predict_batch(
        self,
        image_batch: torch.Tensor,
        return_probability: bool = False
    ) -> Tuple[np.ndarray, list]:
        """
        Predict waterlogging masks for batch of images
        
        Args:
            image_batch: Batch of images (B, C, H, W)
            return_probability: Return probability maps instead of binary masks
            
        Returns:
            Tuple of (masks, metadata_list)
        """
        image_batch = image_batch.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_batch)
            probabilities = torch.sigmoid(logits)
        
        prob_maps = probabilities.squeeze(1).cpu().numpy()
        
        metadata_list = []
        for prob_map in prob_maps:
            metadata = {
                'max_probability': float(prob_map.max()),
                'mean_probability': float(prob_map.mean()),
                'waterlogged_area_ratio': float((prob_map > self.threshold).sum() / prob_map.size),
                'threshold': self.threshold
            }
            metadata_list.append(metadata)
        
        if return_probability:
            return prob_maps, metadata_list
        else:
            binary_masks = (prob_maps > self.threshold).astype(np.uint8)
            return binary_masks, metadata_list
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str, additional_info: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'encoder_name': self.encoder_name,
            'threshold': self.threshold,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def set_threshold(self, threshold: float):
        """Update detection threshold"""
        self.threshold = threshold
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'threshold': self.threshold
        }


class EnsembleDetector:
    """Ensemble of multiple detection models for improved accuracy"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble detector
        
        Args:
            models: List of WaterloggingDetector instances
            weights: Optional weights for each model (must sum to 1)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights
    
    def predict(
        self,
        image_tensor: torch.Tensor,
        return_probability: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Ensemble prediction
        
        Args:
            image_tensor: Input tensor (C, H, W) or (B, C, H, W)
            return_probability: Return probability map instead of binary mask
            
        Returns:
            Tuple of (mask, metadata)
        """
        # Get predictions from all models
        prob_maps = []
        for model in self.models:
            prob_map, _ = model.predict(image_tensor, return_probability=True)
            prob_maps.append(prob_map)
        
        # Weighted average
        ensemble_prob = np.zeros_like(prob_maps[0])
        for prob_map, weight in zip(prob_maps, self.weights):
            ensemble_prob += prob_map * weight
        
        # Calculate metadata
        threshold = np.mean([model.threshold for model in self.models])
        metadata = {
            'max_probability': float(ensemble_prob.max()),
            'mean_probability': float(ensemble_prob.mean()),
            'waterlogged_area_ratio': float((ensemble_prob > threshold).sum() / ensemble_prob.size),
            'threshold': threshold,
            'num_models': len(self.models)
        }
        
        if return_probability:
            return ensemble_prob, metadata
        else:
            binary_mask = (ensemble_prob > threshold).astype(np.uint8)
            return binary_mask, metadata
