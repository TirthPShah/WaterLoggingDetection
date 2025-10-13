"""
Training script for flood detection models.

Supports:
- Custom CNN (trained from scratch)
- Pretrained models (ResNet50, EfficientNet-B0, MobileNetV2)
- Early stopping
- Learning rate scheduling
- Comprehensive metrics tracking
"""

import argparse
import os
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

from src.flood_classifier import create_model
from src.flood_preprocessing import get_flood_detection_transforms


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss improved: {self.best_loss:.6f} → {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def get_summary(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_acc': max(self.val_accs) if self.val_accs else 0,
            'best_val_loss': min(self.val_losses) if self.val_losses else float('inf'),
            'total_time': sum(self.epoch_times)
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device
):
    """
    Main training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
        
    Returns:
        Trained model and training history
    """
    # Setup
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 10),
        min_delta=0.001,
        verbose=True
    )
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Training loop
    best_val_acc = 0.0
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Model: {config['model_name']}")
    print("=" * 70 + "\n")
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        epoch_time = time.time() - epoch_start
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {early_stopping.best_epoch+1}")
            break
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Training Time: {sum(metrics.epoch_times):.2f}s")
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'final_model.pth')
    
    # Save training history
    history = metrics.get_summary()
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Flood Detection Model')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='custom',
                       choices=['custom', 'pretrained'],
                       help='Model type')
    parser.add_argument('--model-name', type=str, default=None,
                       choices=['resnet50', 'efficientnet_b0', 'mobilenet_v2'],
                       help='Pretrained model name (required if model-type=pretrained)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone layers (pretrained only)')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset (ImageFolder format)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='models/flood_classifier',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "=" * 70)
    print("Flood Detection Model Training")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    
    # Create model
    print("\nCreating model...")
    if args.model_type == 'custom':
        model = create_model('custom')
        model_display_name = "Custom CNN"
    else:
        if args.model_name is None:
            raise ValueError("--model-name required for pretrained models")
        model = create_model(
            'pretrained',
            model_name=args.model_name,
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
        model_display_name = f"{args.model_name.upper()} (Pretrained)"
    
    model = model.to(device)
    print(f"✅ Model created: {model_display_name}")
    
    # Load data
    print("\nLoading dataset...")
    train_transform = get_flood_detection_transforms(train=True)
    val_transform = get_flood_detection_transforms(train=False)
    
    # Load full dataset
    full_dataset = ImageFolder(root=args.data_dir, transform=train_transform)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"✅ Dataset loaded")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Classes: {full_dataset.classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training config
    config = {
        'model_type': args.model_type,
        'model_name': args.model_name if args.model_type == 'pretrained' else 'custom_cnn',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'val_split': args.val_split,
        'save_dir': args.save_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Train
    model, history = train_model(model, train_loader, val_loader, config, device)
    
    print("\n✅ Training completed successfully!")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
