"""
Training script for waterlogging detection model
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

import config
from src.detection_model import WaterloggingDetector
from src.preprocessing import AugmentationPipeline


class WaterloggingDataset(Dataset):
    """Dataset for waterlogging detection"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Augmentation transform
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get list of images
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                  list(self.images_dir.glob("*.png")))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask
        
        # Apply transforms
        if self.transform:
            image_tensor, mask_tensor = self.transform.apply_train_transform(image, mask)
            return image_tensor, mask_tensor.unsqueeze(0)
        
        return image, mask


def train_model(
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: str,
    val_masks_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.001
):
    """
    Train detection model
    
    Args:
        train_images_dir: Training images directory
        train_masks_dir: Training masks directory
        val_images_dir: Validation images directory
        val_masks_dir: Validation masks directory
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    print("="*60)
    print("WATERLOGGING DETECTION MODEL TRAINING")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize augmentation pipeline
    augmentation = AugmentationPipeline(
        target_size=config.IMAGE_SIZE,
        normalize_mean=config.NORMALIZE_MEAN,
        normalize_std=config.NORMALIZE_STD
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = WaterloggingDataset(
        train_images_dir,
        train_masks_dir,
        transform=augmentation
    )
    
    val_dataset = WaterloggingDataset(
        val_images_dir,
        val_masks_dir,
        transform=augmentation
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    detector = WaterloggingDetector(
        model_name=config.DETECTION_MODEL_NAME,
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        device=config.DEVICE
    )
    
    model = detector.model
    model.train()
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in train_bar:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks in val_bar:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / "best_model.pth"
            detector.save_checkpoint(
                str(checkpoint_path),
                additional_info={
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
            )
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            detector.save_checkpoint(str(checkpoint_path))
        
        print("-"*60)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Waterlogging Detection Model")
    
    parser.add_argument("--train-images", required=True, help="Training images directory")
    parser.add_argument("--train-masks", required=True, help="Training masks directory")
    parser.add_argument("--val-images", required=True, help="Validation images directory")
    parser.add_argument("--val-masks", required=True, help="Validation masks directory")
    parser.add_argument("--output", default="models/detection", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
