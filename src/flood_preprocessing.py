"""
Custom PyTorch preprocessing pipeline for flood detection from CCTV images.

This module provides a custom transform for cropping the lower portion of images
where road and water accumulation are typically visible.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import random
from typing import Tuple, Optional


class LowerCrop:
    """
    Custom transform to crop the lower portion of an image.
    
    This is particularly useful for CCTV/road images where the key features
    (road surface, water accumulation) are in the lower part of the frame.
    
    Args:
        output_size (int or tuple): Desired output size (height, width).
                                    If int, square crop is made.
        crop_position (float): Position of crop from top (0.0 to 1.0).
                               0.7-0.8 recommended for lower bias.
                               Default: 0.75 (75% down from top)
        random_variation (float): Random variation in crop position (0.0 to 1.0).
                                 Default: 0.05 (±5% variation)
    
    Example:
        >>> transform = LowerCrop(224, crop_position=0.75, random_variation=0.05)
        >>> cropped_img = transform(img)
    """
    
    def __init__(
        self,
        output_size: int,
        crop_position: float = 0.75,
        random_variation: float = 0.05
    ):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        
        self.crop_position = crop_position
        self.random_variation = random_variation
        
        # Validate parameters
        assert 0.0 <= crop_position <= 1.0, "crop_position must be between 0 and 1"
        assert 0.0 <= random_variation <= 1.0, "random_variation must be between 0 and 1"
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply lower crop to image.
        
        Args:
            img: PIL Image
            
        Returns:
            Cropped PIL Image
        """
        w, h = img.size
        crop_h, crop_w = self.output_size
        
        # Ensure crop size fits in image
        if h < crop_h or w < crop_w:
            raise ValueError(
                f"Image size ({w}x{h}) is smaller than crop size ({crop_w}x{crop_h})"
            )
        
        # Calculate crop position with random variation
        variation = random.uniform(
            -self.random_variation,
            self.random_variation
        )
        adjusted_position = self.crop_position + variation
        adjusted_position = max(0.0, min(1.0, adjusted_position))
        
        # Calculate top-left corner of crop
        # Position determines where the CENTER of the crop will be vertically
        center_y = int(h * adjusted_position)
        
        # Calculate actual top coordinate (ensuring crop stays within bounds)
        top = center_y - crop_h // 2
        top = max(0, min(top, h - crop_h))
        
        # Center horizontally
        left = (w - crop_w) // 2
        left = max(0, min(left, w - crop_w))
        
        # Perform crop
        cropped_img = img.crop((left, top, left + crop_w, top + crop_h))
        
        return cropped_img
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}, "
            f"crop_position={self.crop_position}, "
            f"random_variation={self.random_variation})"
        )


def get_flood_detection_transforms(
    train: bool = True,
    resize_size: int = 488,
    crop_size: int = 244,
    crop_position: float = 0.85,
    random_variation: float = 0.05
) -> transforms.Compose:
    """
    Get preprocessing transforms for flood detection images.
    
    Pipeline:
    1. Resize image so shorter side = resize_size (maintains aspect ratio)
    2. Crop bottom middle portion of image (244x244) where road/water is visible
    3. Optional: Random horizontal flip (training only)
    4. Convert to tensor
    5. Normalize using ImageNet statistics
    
    Args:
        train: If True, applies data augmentation (horizontal flip)
        resize_size: Size of shorter side after resize (default: 488)
        crop_size: Size of square crop (default: 244)
        crop_position: Vertical position for crop (0.85 = 85% down from top for bottom)
        random_variation: Random variation in crop position (default: 0.05)
    
    Returns:
        transforms.Compose object with all transformations
    """
    
    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    transform_list = []
    
    # Step 1: Resize maintaining aspect ratio (shorter side = resize_size)
    transform_list.append(
        transforms.Resize(resize_size)
    )
    
    # Step 2: Crop lower portion of image
    transform_list.append(
        LowerCrop(
            output_size=crop_size,
            crop_position=crop_position,
            random_variation=random_variation if train else 0.0
        )
    )
    
    # Step 3: Data augmentation (training only)
    if train:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )
        # Optional: Add color jitter for robustness
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            )
        )
    
    # Step 4: Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Step 5: Normalize using ImageNet statistics
    transform_list.append(
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )
    
    return transforms.Compose(transform_list)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet statistics for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W)
        
    Returns:
        Denormalized tensor
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    denorm_tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    return denorm_tensor


class ResizeAndShowCrop:
    """
    Helper transform to resize and show where crop will be taken.
    Used for visualization purposes only.
    """
    
    def __init__(
        self,
        resize_size: int = 488,
        crop_size: int = 244,
        crop_position: float = 0.85
    ):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.crop_position = crop_position
    
    def __call__(self, img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Resize image and return crop coordinates.
        
        Returns:
            Tuple of (resized_image, (left, top, right, bottom))
        """
        # Resize
        resized = transforms.Resize(self.resize_size)(img)
        
        w, h = resized.size
        crop_h, crop_w = self.crop_size, self.crop_size
        
        # Calculate crop position
        center_y = int(h * self.crop_position)
        top = center_y - crop_h // 2
        top = max(0, min(top, h - crop_h))
        
        left = (w - crop_w) // 2
        left = max(0, min(left, w - crop_w))
        
        crop_box = (left, top, left + crop_w, top + crop_h)
        
        return resized, crop_box
