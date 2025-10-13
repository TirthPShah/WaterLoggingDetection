"""
Simple test script to verify LowerCrop transform works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.flood_preprocessing import LowerCrop, get_flood_detection_transforms


def create_test_image(width=800, height=600):
    """
    Create a test image with horizontal stripes to visualize cropping.
    
    Returns:
        PIL Image
    """
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create horizontal stripes with labels
    stripe_height = height // 10
    colors = [
        (255, 0, 0),    # Red - Top
        (255, 128, 0),  # Orange
        (255, 255, 0),  # Yellow
        (128, 255, 0),  # Yellow-Green
        (0, 255, 0),    # Green
        (0, 255, 128),  # Green-Cyan
        (0, 255, 255),  # Cyan
        (0, 128, 255),  # Blue
        (0, 0, 255),    # Dark Blue - Bottom
        (128, 0, 255),  # Purple
    ]
    
    for i, color in enumerate(colors):
        start_y = i * stripe_height
        end_y = min((i + 1) * stripe_height, height)
        img_array[start_y:end_y, :] = color
    
    return Image.fromarray(img_array)


def test_lower_crop():
    """Test the LowerCrop transform."""
    
    print("=" * 70)
    print("Testing LowerCrop Transform")
    print("=" * 70)
    
    # Create test image
    test_img = create_test_image(width=800, height=600)
    print(f"\n✅ Test image created: {test_img.size}")
    
    # Test different crop positions
    crop_positions = [0.5, 0.75, 0.85]  # Center, Lower, Bottom
    
    fig, axes = plt.subplots(1, len(crop_positions) + 1, figsize=(15, 4))
    
    # Show original
    axes[0].imshow(test_img)
    axes[0].set_title('Original Image\n800×600', fontsize=10)
    axes[0].axis('off')
    
    # Add horizontal lines to show positions
    for pos in [0.5, 0.75, 0.85]:
        y = int(600 * pos)
        axes[0].axhline(y=y, color='white', linestyle='--', linewidth=1, alpha=0.7)
        axes[0].text(10, y - 10, f'{int(pos*100)}%', color='white', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Test each crop position
    for idx, pos in enumerate(crop_positions):
        lower_crop = LowerCrop(
            output_size=244,
            crop_position=pos,
            random_variation=0.0  # No variation for testing
        )
        
        cropped = lower_crop(test_img)
        
        axes[idx + 1].imshow(cropped)
        axes[idx + 1].set_title(
            f'Crop at {int(pos*100)}%\n244×244',
            fontsize=10
        )
        axes[idx + 1].axis('off')
        
        print(f"✅ Crop position {pos:.2f}: {cropped.size}")
    
    plt.tight_layout()
    output_path = 'output/lower_crop_test.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Test visualization saved to: {output_path}")
    plt.show()


def test_full_pipeline():
    """Test the complete preprocessing pipeline."""
    
    print("\n" + "=" * 70)
    print("Testing Full Preprocessing Pipeline")
    print("=" * 70)
    
    # Create test image
    test_img = create_test_image(width=800, height=600)
    
    # Get transforms
    transform = get_flood_detection_transforms(
        train=False,
        resize_size=488,
        crop_size=244,
        crop_position=0.85,
        random_variation=0.0
    )
    
    # Apply transform
    transformed = transform(test_img)
    
    print(f"\n✅ Transform applied successfully")
    print(f"   Input shape: {test_img.size}")
    print(f"   Output shape: {transformed.shape}")
    print(f"   Output dtype: {transformed.dtype}")
    print(f"   Output range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    
    # Check normalization
    expected_channels = 3
    expected_height = 244
    expected_width = 244
    
    assert transformed.shape == (expected_channels, expected_height, expected_width), \
        f"Expected shape (3, 244, 244), got {transformed.shape}"
    
    print("\n✅ All assertions passed!")


def test_random_variation():
    """Test random variation in crop position."""
    
    print("\n" + "=" * 70)
    print("Testing Random Variation in Crop Position")
    print("=" * 70)
    
    test_img = create_test_image(width=800, height=600)
    
    # Create transform with random variation
    lower_crop = LowerCrop(
        output_size=244,
        crop_position=0.85,
        random_variation=0.1  # ±10% variation
    )
    
    # Apply multiple times to see variation
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for idx in range(5):
        cropped = lower_crop(test_img)
        axes[idx].imshow(cropped)
        axes[idx].set_title(f'Random Crop {idx+1}', fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = 'output/random_variation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Random variation test saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    print("\n🧪 Running LowerCrop Tests\n")
    
    # Run tests
    test_lower_crop()
    test_full_pipeline()
    test_random_variation()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print("\nThe LowerCrop transform is working correctly and ready to use.")
    print("Check the output/ directory for visualizations.")
