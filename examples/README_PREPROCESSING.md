# Flood Detection Preprocessing Pipeline

## Overview

This preprocessing pipeline is specifically designed for flood detection from CCTV images, with a focus on capturing the **lower portion of images** where roads and water accumulation are typically visible.

## Key Features

✅ **LowerCrop Transform**: Custom PyTorch transform that crops the lower 70-80% of images  
✅ **Aspect Ratio Preservation**: Resizes images intelligently before cropping  
✅ **Augmentation Support**: Random variation in crop position for training  
✅ **ImageNet Normalization**: Ready for transfer learning  
✅ **Easy Integration**: Works seamlessly with `torchvision.datasets.ImageFolder`

## Quick Start

### 1. Basic Usage

```python
from src.flood_preprocessing import get_flood_detection_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Get preprocessing transforms
transforms = get_flood_detection_transforms(
    train=True,          # Enable augmentation
    resize_size=256,     # Shorter side → 256px
    crop_size=224,       # Final crop size
    crop_position=0.75,  # 75% down from top
    random_variation=0.05 # ±5% random variation
)

# Create dataset
dataset = ImageFolder(root='data/train', transform=transforms)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Use in training loop
for images, labels in dataloader:
    # images shape: (batch_size, 3, 224, 224)
    # Ready for model input!
    pass
```

### 2. Custom LowerCrop Only

```python
from src.flood_preprocessing import LowerCrop
from PIL import Image

# Create transform
lower_crop = LowerCrop(
    output_size=224,
    crop_position=0.75,    # Center of crop at 75% from top
    random_variation=0.05  # ±5% random shift
)

# Apply to image
img = Image.open('photo.jpg')
cropped = lower_crop(img)  # Returns 224×224 PIL Image
```

## Pipeline Steps

The complete preprocessing pipeline performs:

1. **Resize**: Shorter side → 256px (maintains aspect ratio)
2. **LowerCrop**: Extract 224×224 patch from lower portion
3. **Augmentation** (training only):
   - Random horizontal flip (50%)
   - Color jitter (brightness, contrast, saturation, hue)
4. **ToTensor**: Convert PIL Image → PyTorch tensor
5. **Normalize**: Apply ImageNet statistics

```
Input: Variable size image (e.g., 1920×1080)
   ↓ Resize (shorter side = 256)
Resized: (456×256) [aspect ratio preserved]
   ↓ LowerCrop (75% down, 224×224)
Cropped: (224×224) [lower portion with road/water]
   ↓ Augmentation (optional)
Augmented: (224×224) [flipped, color adjusted]
   ↓ ToTensor + Normalize
Output: (3, 224, 224) tensor [-3.0 to +3.0 range]
```

## Why Lower Crop?

CCTV and road surveillance images typically have:
- **Sky and buildings** in the upper portion (less relevant)
- **Road surface and water** in the lower portion (critical for detection)

Standard center crop would miss important features!

### Comparison

```
┌─────────────────┐
│      Sky        │  ← Upper 30% (less important)
│   Buildings     │
├─────────────────┤
│     Road        │  ← Middle 40% (some features)
│   Vehicles      │
├─────────────────┤
│ Road Surface    │  ← Lower 30% (CRITICAL!)
│  Water Pool     │  ← This is what we want to capture
└─────────────────┘

Center Crop: Captures middle → Misses water
Lower Crop:  Captures bottom → Captures water ✓
```

## Examples

### Example 1: Running Tests

```bash
# Test the LowerCrop transform
python examples/test_lower_crop.py
```

This creates visualizations showing:
- How different crop positions work
- Random variation in action
- Full pipeline output

### Example 2: Full Dataset Example

```bash
# Run complete example with your data
python examples/flood_data_pipeline_example.py
```

This will:
- Load your dataset from `data/train/`
- Create DataLoader
- Visualize 5 random samples (before/after)
- Save visualization to `output/preprocessing_visualization.png`

### Example 3: Integration with Training

```python
from src.flood_preprocessing import get_flood_detection_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

# Create datasets
train_transforms = get_flood_detection_transforms(train=True)
val_transforms = get_flood_detection_transforms(train=False)

train_dataset = ImageFolder('data/train', transform=train_transforms)
val_dataset = ImageFolder('data/val', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained model (works with ImageNet normalization)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary: flooded/non-flooded

# Train
for images, labels in train_loader:
    outputs = model(images)
    # ... rest of training loop
```

## API Reference

### `LowerCrop`

Custom transform to crop the lower portion of an image.

**Parameters:**
- `output_size` (int or tuple): Desired output size. If int, makes square crop.
- `crop_position` (float, default=0.75): Vertical position of crop center (0.0-1.0)
  - 0.5 = center crop
  - 0.75 = lower crop (recommended for road images)
  - 0.9 = very low crop
- `random_variation` (float, default=0.05): Random variation in position (0.0-1.0)

**Example:**
```python
transform = LowerCrop(224, crop_position=0.75, random_variation=0.05)
```

### `get_flood_detection_transforms()`

Get complete preprocessing pipeline.

**Parameters:**
- `train` (bool, default=True): Enable augmentation
- `resize_size` (int, default=256): Size of shorter side after resize
- `crop_size` (int, default=224): Size of square crop
- `crop_position` (float, default=0.75): Vertical crop position
- `random_variation` (float, default=0.05): Random variation for augmentation

**Returns:** `transforms.Compose` object

**Example:**
```python
train_transform = get_flood_detection_transforms(train=True)
val_transform = get_flood_detection_transforms(train=False)
```

### `denormalize()`

Denormalize tensor for visualization.

**Parameters:**
- `tensor` (torch.Tensor): Normalized tensor (C, H, W)

**Returns:** Denormalized tensor (0-1 range)

**Example:**
```python
from src.flood_preprocessing import denormalize
import matplotlib.pyplot as plt

# After getting normalized tensor from dataloader
denorm_image = denormalize(tensor)
plt.imshow(denorm_image.permute(1, 2, 0))
```

## Data Organization

Organize your data in ImageFolder format:

```
data/
├── train/
│   ├── flooded/
│   │   ├── flood_001.jpg
│   │   ├── flood_002.jpg
│   │   └── ...
│   └── non_flooded/
│       ├── normal_001.jpg
│       ├── normal_002.jpg
│       └── ...
├── val/
│   ├── flooded/
│   └── non_flooded/
└── test/
    ├── flooded/
    └── non_flooded/
```

## Customization

### Adjust Crop Position

For different camera angles:

```python
# High-angle camera (more road visible in upper portion)
transforms = get_flood_detection_transforms(crop_position=0.6)

# Low-angle camera (road in lower portion)
transforms = get_flood_detection_transforms(crop_position=0.85)
```

### Larger Input Size

For higher resolution models:

```python
transforms = get_flood_detection_transforms(
    resize_size=512,  # Resize to 512
    crop_size=448     # Crop 448×448
)
```

### Disable Augmentation

For validation/testing:

```python
val_transform = get_flood_detection_transforms(
    train=False,           # Disable augmentation
    random_variation=0.0   # Deterministic crop
)
```

## Troubleshooting

### Issue: "Image size is smaller than crop size"

**Solution:** Your images are too small. Either:
- Reduce `crop_size` parameter
- Increase `resize_size` parameter
- Check your image files aren't corrupted

### Issue: Crop doesn't capture road

**Solution:** Adjust `crop_position`:
- Increase if road is lower (try 0.8 or 0.85)
- Decrease if road is higher (try 0.65 or 0.7)

### Issue: Colors look wrong after transform

**Solution:** This is normal! Images are normalized. Use `denormalize()` for visualization:

```python
from src.flood_preprocessing import denormalize
denorm = denormalize(tensor)
plt.imshow(denorm.permute(1, 2, 0))
```

## Performance Tips

1. **Use `pin_memory=True`** in DataLoader for faster GPU transfer
2. **Use `num_workers > 0`** for parallel data loading
3. **Pre-resize images offline** if dataset is large (one-time cost)
4. **Cache on SSD** for faster I/O

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@software{flood_preprocessing_2025,
  title = {Lower Crop Preprocessing for Flood Detection},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/TirthPShah/WaterLoggingDetection}
}
```

## License

MIT License - See LICENSE file for details.
