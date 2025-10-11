"""
Utility script to create sample dataset for testing and demonstration
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def generate_sample_image_with_water(
    size: tuple = (512, 512),
    water_ratio: float = 0.3,
    seed: int = None
) -> tuple:
    """
    Generate synthetic road image with waterlogged regions
    
    Args:
        size: Image size (height, width)
        water_ratio: Ratio of waterlogged area (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (image, mask)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = size
    
    # Create base road image (gray with texture)
    image = np.random.randint(60, 120, (h, w, 3), dtype=np.uint8)
    
    # Add road texture
    for _ in range(100):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(image, (x, y), np.random.randint(1, 3), 
                  (np.random.randint(50, 130),) * 3, -1)
    
    # Create road markings
    cv2.line(image, (w//4, 0), (w//4, h), (200, 200, 200), 5)
    cv2.line(image, (3*w//4, 0), (3*w//4, h), (200, 200, 200), 5)
    
    # Create water mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add random water puddles
    num_puddles = np.random.randint(3, 8)
    target_area = int(h * w * water_ratio)
    current_area = 0
    
    for _ in range(num_puddles):
        if current_area >= target_area:
            break
        
        # Random puddle position and size
        center_x = np.random.randint(w//4, 3*w//4)
        center_y = np.random.randint(h//4, 3*h//4)
        size_x = np.random.randint(30, 100)
        size_y = np.random.randint(30, 100)
        
        # Draw ellipse for puddle
        cv2.ellipse(mask, (center_x, center_y), (size_x, size_y),
                   np.random.randint(0, 180), 0, 360, 255, -1)
        
        # Add water appearance to image
        water_overlay = image.copy()
        water_color = (np.random.randint(100, 150), 
                      np.random.randint(120, 180), 
                      np.random.randint(80, 130))
        cv2.ellipse(water_overlay, (center_x, center_y), (size_x, size_y),
                   np.random.randint(0, 180), 0, 360, water_color, -1)
        
        # Blend water with road
        water_mask = (mask > 0).astype(np.uint8)
        image = cv2.addWeighted(image, 0.6, water_overlay, 0.4, 0)
        
        current_area += cv2.countNonZero(mask)
    
    # Add some noise and blur to make it more realistic
    image = cv2.GaussianBlur(image, (5, 5), 0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    
    return image, mask


def create_sample_dataset(
    output_dir: str,
    num_images: int = 100,
    train_ratio: float = 0.8,
    image_size: tuple = (512, 512)
):
    """
    Create sample dataset with synthetic images
    
    Args:
        output_dir: Output directory for dataset
        num_images: Total number of images to generate
        train_ratio: Ratio of training images
        image_size: Image size
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    num_train = int(num_images * train_ratio)
    num_val = num_images - num_train
    
    print(f"Creating sample dataset: {num_train} train, {num_val} val images")
    
    # Generate training images
    print("\nGenerating training images...")
    for i in range(num_train):
        water_ratio = np.random.uniform(0.05, 0.5)
        image, mask = generate_sample_image_with_water(
            size=image_size,
            water_ratio=water_ratio,
            seed=i
        )
        
        cv2.imwrite(str(train_dir / "images" / f"image_{i:04d}.jpg"), image)
        cv2.imwrite(str(train_dir / "masks" / f"image_{i:04d}.png"), mask)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_train} training images")
    
    # Generate validation images
    print("\nGenerating validation images...")
    for i in range(num_val):
        water_ratio = np.random.uniform(0.05, 0.5)
        image, mask = generate_sample_image_with_water(
            size=image_size,
            water_ratio=water_ratio,
            seed=num_train + i
        )
        
        cv2.imwrite(str(val_dir / "images" / f"image_{i:04d}.jpg"), image)
        cv2.imwrite(str(val_dir / "masks" / f"image_{i:04d}.png"), mask)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_val} validation images")
    
    print(f"\n✅ Dataset created successfully at: {output_dir}")
    print(f"   Train: {num_train} images")
    print(f"   Val: {num_val} images")


def main():
    parser = argparse.ArgumentParser(
        description="Create sample dataset for waterlogging detection"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Total number of images to generate"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training images (0-1)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size (height width)"
    )
    
    args = parser.parse_args()
    
    create_sample_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        train_ratio=args.train_ratio,
        image_size=tuple(args.image_size)
    )


if __name__ == "__main__":
    main()
