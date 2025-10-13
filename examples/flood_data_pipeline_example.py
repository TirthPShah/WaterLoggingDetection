"""
Example usage of the flood detection preprocessing pipeline.

This script demonstrates:
1. Creating a dataset with custom transforms
2. Setting up DataLoader for training
3. Visualizing the preprocessing results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from src.flood_preprocessing import (
    get_flood_detection_transforms,
    LowerCrop,
    denormalize,
    ResizeAndShowCrop
)


def create_flood_dataset(
    data_dir: str,
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Create dataset and dataloader for flood detection.
    
    Args:
        data_dir: Path to dataset organized in ImageFolder format
                  (e.g., data/train/ with subdirectories for each class)
        train: If True, applies training augmentations
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    
    # Get transforms
    transforms = get_flood_detection_transforms(
        train=train,
        resize_size=488,
        crop_size=244,
        crop_position=0.85,  # 85% down from top (bottom portion)
        random_variation=0.05  # ±5% variation
    )
    
    # Create dataset
    dataset = ImageFolder(
        root=data_dir,
        transform=transforms
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    return dataset, dataloader


def visualize_preprocessing(
    data_dir: str,
    num_samples: int = 5,
    save_path: str = 'output/preprocessing_visualization.png'
):
    """
    Visualize the preprocessing pipeline on random samples.
    
    Shows before (resized + crop box) and after (final transformed) images.
    
    Args:
        data_dir: Path to image directory
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get dataset without transforms to access raw images
    dataset_raw = ImageFolder(root=data_dir)
    
    # Get transforms for final output
    transforms_final = get_flood_detection_transforms(
        train=False,  # No random augmentation for visualization
        resize_size=488,
        crop_size=244,
        crop_position=0.85,
        random_variation=0.0
    )
    
    # Helper to show crop box
    resize_and_crop = ResizeAndShowCrop(
        resize_size=488,
        crop_size=244,
        crop_position=0.85
    )
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Randomly sample images
    indices = np.random.choice(len(dataset_raw), num_samples, replace=False)
    
    for idx, img_idx in enumerate(indices):
        # Get original image and label
        img, label = dataset_raw[img_idx]
        class_name = dataset_raw.classes[label]
        
        # Left plot: Resized image with crop box
        resized_img, crop_box = resize_and_crop(img)
        left, top, right, bottom = crop_box
        
        axes[idx, 0].imshow(resized_img)
        axes[idx, 0].set_title(
            f'Original (Resized)\nClass: {class_name}\nSize: {resized_img.size}',
            fontsize=10
        )
        axes[idx, 0].axis('off')
        
        # Draw crop box
        rect = patches.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label='Crop Area'
        )
        axes[idx, 0].add_patch(rect)
        
        # Add text indicating crop position
        axes[idx, 0].text(
            left + 5,
            top + 15,
            'Bottom Crop\n(85% down)',
            color='red',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Right plot: Final transformed image
        transformed = transforms_final(img)
        
        # Denormalize for visualization
        denorm = denormalize(transformed)
        denorm_np = denorm.permute(1, 2, 0).numpy()
        
        axes[idx, 1].imshow(denorm_np)
        axes[idx, 1].set_title(
            f'After Transform\nClass: {class_name}\nSize: 244x244',
            fontsize=10
        )
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.show()


def main():
    """
    Main example demonstrating the preprocessing pipeline.
    """
    
    print("=" * 70)
    print("Flood Detection Preprocessing Pipeline - Example Usage")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = "data/train"  # Adjust path as needed
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\n⚠️  Data directory not found: {DATA_DIR}")
        print("\nPlease organize your data in ImageFolder format:")
        print("  data/train/")
        print("    ├── flooded/")
        print("    │   ├── img1.jpg")
        print("    │   ├── img2.jpg")
        print("    └── non_flooded/")
        print("        ├── img1.jpg")
        print("        ├── img2.jpg")
        print("\nOr use data/raw if you have images there.")
        
        # Try data/raw as fallback
        DATA_DIR = "data/raw"
        if not Path(DATA_DIR).exists():
            print(f"\n❌ {DATA_DIR} also not found. Please add images first.")
            return
        else:
            print(f"\n✅ Using fallback: {DATA_DIR}")
    
    print(f"\n📁 Data Directory: {DATA_DIR}")
    
    # Create dataset and dataloader
    print("\n" + "-" * 70)
    print("Creating Dataset and DataLoader")
    print("-" * 70)
    
    try:
        train_dataset, train_loader = create_flood_dataset(
            data_dir=DATA_DIR,
            train=True,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Total images: {len(train_dataset)}")
        print(f"   Classes: {train_dataset.classes}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Number of batches: {len(train_loader)}")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        print("\nMake sure your data is organized in ImageFolder format.")
        return
    
    # Test batch loading
    print("\n" + "-" * 70)
    print("Testing Batch Loading")
    print("-" * 70)
    
    try:
        images, labels = next(iter(train_loader))
        print(f"✅ Batch loaded successfully")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        return
    
    # Visualize preprocessing
    print("\n" + "-" * 70)
    print("Visualizing Preprocessing Pipeline")
    print("-" * 70)
    
    try:
        visualize_preprocessing(
            data_dir=DATA_DIR,
            num_samples=5,
            save_path='output/preprocessing_visualization.png'
        )
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Show example of custom LowerCrop usage
    print("\n" + "-" * 70)
    print("Example: Custom LowerCrop Transform")
    print("-" * 70)
    
    lower_crop = LowerCrop(
        output_size=224,
        crop_position=0.75,
        random_variation=0.05
    )
    print(f"Transform: {lower_crop}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print("""
Preprocessing Steps:
1. ✅ Resize: Shorter side → 488px (maintains aspect ratio)
2. ✅ LowerCrop: Crop 244×244 at 85% down from top (bottom middle, ±5% variation)
3. ✅ Augmentation: Random horizontal flip + color jitter (training only)
4. ✅ ToTensor: Convert to PyTorch tensor
5. ✅ Normalize: ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

Key Features:
- Custom LowerCrop focuses on road/water in bottom of image
- Maintains aspect ratio before cropping
- Random variation for augmentation
- Ready for transfer learning with ImageNet-pretrained models
    """)
    
    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
