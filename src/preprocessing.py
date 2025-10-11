"""
Image preprocessing module for CCTV frames
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import torch


class ImagePreprocessor:
    """Preprocess CCTV images for waterlogging detection"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        apply_clahe: bool = True,
        apply_denoising: bool = False
    ):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: Mean for normalization (RGB)
            normalize_std: Std for normalization (RGB)
            apply_clahe: Apply CLAHE for contrast enhancement
            apply_denoising: Apply denoising
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.apply_clahe = apply_clahe
        self.apply_denoising = apply_denoising
        
        # Create albumentations transform pipeline
        self.transform = self._create_transform()
        
        # CLAHE for contrast enhancement
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def _create_transform(self) -> A.Compose:
        """Create albumentations transformation pipeline"""
        transforms = [
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ]
        return A.Compose(transforms)
    
    def preprocess(
        self, 
        image: np.ndarray,
        adjust_brightness: bool = True,
        adjust_contrast: bool = True
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess a single image
        
        Args:
            image: Input image in BGR format (H, W, C)
            adjust_brightness: Apply brightness adjustment
            adjust_contrast: Apply contrast adjustment
            
        Returns:
            Tuple of (preprocessed_tensor, preprocessed_image)
            - preprocessed_tensor: Torch tensor ready for model input
            - preprocessed_image: Numpy array for visualization
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        processed = image_rgb.copy()
        
        # Denoising
        if self.apply_denoising:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Brightness and contrast adjustment
        if adjust_brightness or adjust_contrast:
            processed = self._adjust_brightness_contrast(
                processed, 
                apply_brightness=adjust_brightness,
                apply_contrast=adjust_contrast
            )
        
        # CLAHE for contrast enhancement
        if self.apply_clahe:
            processed = self._apply_clahe(processed)
        
        # Apply albumentations transforms
        augmented = self.transform(image=processed)
        tensor = augmented['image']
        
        return tensor, processed
    
    def _adjust_brightness_contrast(
        self,
        image: np.ndarray,
        apply_brightness: bool = True,
        apply_contrast: bool = True,
        brightness_beta: Optional[float] = None,
        contrast_alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image (RGB)
            apply_brightness: Apply brightness adjustment
            apply_contrast: Apply contrast adjustment
            brightness_beta: Manual brightness value (auto-calculated if None)
            contrast_alpha: Manual contrast value (auto-calculated if None)
            
        Returns:
            Adjusted image
        """
        # Auto-calculate brightness and contrast if not provided
        if brightness_beta is None and apply_brightness:
            # Calculate average brightness
            brightness = np.mean(image)
            target_brightness = 127  # Target middle brightness
            brightness_beta = target_brightness - brightness
            brightness_beta = np.clip(brightness_beta, -50, 50)
        else:
            brightness_beta = brightness_beta or 0
        
        if contrast_alpha is None and apply_contrast:
            # Auto-adjust contrast based on histogram spread
            std_dev = np.std(image)
            if std_dev < 40:
                contrast_alpha = 1.3  # Increase contrast for low-contrast images
            elif std_dev > 80:
                contrast_alpha = 0.9  # Decrease contrast for high-contrast images
            else:
                contrast_alpha = 1.0
        else:
            contrast_alpha = contrast_alpha or 1.0
        
        # Apply brightness and contrast adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=contrast_alpha, beta=brightness_beta)
        
        return adjusted
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def preprocess_batch(
        self,
        images: list,
        adjust_brightness: bool = True,
        adjust_contrast: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """
        Preprocess a batch of images
        
        Args:
            images: List of images in BGR format
            adjust_brightness: Apply brightness adjustment
            adjust_contrast: Apply contrast adjustment
            
        Returns:
            Tuple of (batch_tensor, preprocessed_images)
        """
        tensors = []
        processed_images = []
        
        for image in images:
            tensor, processed = self.preprocess(
                image,
                adjust_brightness=adjust_brightness,
                adjust_contrast=adjust_contrast
            )
            tensors.append(tensor)
            processed_images.append(processed)
        
        # Stack tensors into batch
        batch_tensor = torch.stack(tensors)
        
        return batch_tensor, processed_images
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor back to image
        
        Args:
            tensor: Normalized tensor (C, H, W)
            
        Returns:
            Image array (H, W, C) in RGB format
        """
        # Convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        image = tensor.numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array(self.normalize_mean)
        std = np.array(self.normalize_std)
        image = image * std + mean
        
        # Clip to valid range
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image


class AugmentationPipeline:
    """Data augmentation pipeline for training"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize augmentation pipeline
        
        Args:
            target_size: Target image size
            normalize_mean: Mean for normalization
            normalize_std: Std for normalization
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
    
    def _create_train_transform(self) -> A.Compose:
        """Create training augmentation pipeline"""
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ])
    
    def _create_val_transform(self) -> A.Compose:
        """Create validation transformation pipeline"""
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ])
    
    def apply_train_transform(self, image: np.ndarray, mask: np.ndarray = None):
        """Apply training augmentation"""
        if mask is not None:
            augmented = self.train_transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.train_transform(image=image)
            return augmented['image']
    
    def apply_val_transform(self, image: np.ndarray, mask: np.ndarray = None):
        """Apply validation transformation"""
        if mask is not None:
            augmented = self.val_transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.val_transform(image=image)
            return augmented['image']
