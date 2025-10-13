"""
Utility script to prepare and preprocess downloaded flood detection datasets
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm


def resize_image(image_path: Path, output_path: Path, size: tuple = (512, 512)):
    """
    Resize image to specified size
    
    Args:
        image_path: Path to input image
        output_path: Path to save resized image
        size: Target size (height, width)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return False
    
    resized = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(output_path), resized)
    return True


def resize_mask(mask_path: Path, output_path: Path, size: tuple = (512, 512)):
    """
    Resize mask to specified size (using nearest neighbor to preserve labels)
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save resized mask
        size: Target size (height, width)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read {mask_path}")
        return False
    
    # Binarize mask (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Resize using nearest neighbor to preserve binary values
    resized = cv2.resize(binary_mask, (size[1], size[0]), 
                        interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(str(output_path), resized)
    return True


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    size: tuple = (512, 512),
    image_exts: list = ['.jpg', '.jpeg', '.png'],
    mask_exts: list = ['.png', '.jpg']
):
    """
    Prepare dataset by resizing all images and masks
    
    Args:
        input_dir: Input directory containing images and masks
        output_dir: Output directory for processed dataset
        size: Target image size (height, width)
        image_exts: Valid image file extensions
        mask_exts: Valid mask file extensions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find images and masks subdirectories
    image_dirs = list(input_path.rglob("*images*"))
    mask_dirs = list(input_path.rglob("*masks*")) + list(input_path.rglob("*labels*"))
    
    if not image_dirs:
        print(f"No 'images' directory found in {input_dir}")
        # Try to use input_dir directly
        image_dirs = [input_path]
    
    for img_dir in image_dirs:
        print(f"\nProcessing images from: {img_dir}")
        
        # Determine if this is train/val/test
        split_name = None
        for split in ['train', 'val', 'test', 'validation']:
            if split in str(img_dir).lower():
                split_name = split if split != 'validation' else 'val'
                break
        
        if split_name is None:
            split_name = 'processed'
        
        # Create output directories
        output_img_dir = output_path / split_name / 'images'
        output_mask_dir = output_path / split_name / 'masks'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        image_files = []
        for ext in image_exts:
            image_files.extend(img_dir.glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images")
        
        success_count = 0
        for img_path in tqdm(image_files, desc="Resizing images"):
            output_img_path = output_img_dir / img_path.name
            if resize_image(img_path, output_img_path, size):
                success_count += 1
        
        print(f"Successfully resized {success_count}/{len(image_files)} images")
        
        # Find corresponding mask directory
        mask_dir = None
        for mdir in mask_dirs:
            if split_name in str(mdir).lower() or split_name == 'processed':
                mask_dir = mdir
                break
        
        if mask_dir and mask_dir.exists():
            print(f"\nProcessing masks from: {mask_dir}")
            
            # Process masks
            mask_files = []
            for ext in mask_exts:
                mask_files.extend(mask_dir.glob(f"*{ext}"))
            
            print(f"Found {len(mask_files)} masks")
            
            success_count = 0
            for mask_path in tqdm(mask_files, desc="Resizing masks"):
                output_mask_path = output_mask_dir / mask_path.name
                if resize_mask(mask_path, output_mask_path, size):
                    success_count += 1
            
            print(f"Successfully resized {success_count}/{len(mask_files)} masks")
        else:
            print(f"Warning: No mask directory found for {split_name}")
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"Output directory: {output_path}")


def split_dataset(
    input_dir: str,
    output_dir: str,
    split_ratios: list = [0.7, 0.2, 0.1],
    seed: int = 42
):
    """
    Split dataset into train/val/test sets
    
    Args:
        input_dir: Input directory with all images and masks
        output_dir: Output directory for split dataset
        split_ratios: [train, val, test] ratios
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all images
    image_dir = input_path / 'images'
    mask_dir = input_path / 'masks'
    
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        return
    
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"Found {len(image_files)} images")
    
    # Shuffle
    indices = np.random.permutation(len(image_files))
    
    # Calculate split sizes
    train_size = int(len(image_files) * split_ratios[0])
    val_size = int(len(image_files) * split_ratios[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Create directories and copy files
    for split_name, split_indices in splits.items():
        split_img_dir = output_path / split_name / 'images'
        split_mask_dir = output_path / split_name / 'masks'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{split_name}: {len(split_indices)} images")
        
        for idx in tqdm(split_indices, desc=f"Copying {split_name}"):
            img_file = image_files[idx]
            
            # Copy image
            shutil.copy(img_file, split_img_dir / img_file.name)
            
            # Copy mask if exists
            mask_file = mask_dir / img_file.name
            if not mask_file.exists():
                # Try with .png extension
                mask_file = mask_dir / (img_file.stem + '.png')
            
            if mask_file.exists():
                shutil.copy(mask_file, split_mask_dir / mask_file.name)
    
    print(f"\n✅ Dataset split complete!")
    print(f"Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess flood detection dataset"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Target image size (height width)'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split dataset into train/val/test'
    )
    parser.add_argument(
        '--split-ratios',
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help='Train/val/test split ratios'
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        size=tuple(args.size)
    )
    
    # Split if requested
    if args.split:
        split_dataset(
            input_dir=args.output,
            output_dir='data',
            split_ratios=args.split_ratios
        )


if __name__ == "__main__":
    main()
