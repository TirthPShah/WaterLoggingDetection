# Flood Detection Classification Project - Complete Guide

## 📋 Project Overview

A comprehensive computer vision system for binary flood detection (Flooded vs Non-Flooded) using:
- **Custom CNN** trained from scratch
- **Pretrained models** (ResNet50, EfficientNet-B0, MobileNetV2) with fine-tuning
- Complete evaluation framework with Grad-CAM visualization
- Model comparison and benchmarking tools

## 🎯 Objectives

1. Develop and compare custom CNN vs pretrained models
2. Achieve high accuracy on balanced dataset (600 flooded + 600 non-flooded images)
3. Analyze trade-offs between accuracy, speed, and model size
4. Provide interpretability through Grad-CAM visualizations

## 📊 Dataset Specifications

- **Total Images**: 1,200 (600 per class)
- **Resolution**: 244×244×3 (RGB)
- **Split**: 80% training, 20% validation
- **Classes**: Binary (0=Non-Flooded, 1=Flooded)
- **Format**: ImageFolder structure

### Expected Directory Structure

```
data/
├── train/
│   ├── flooded/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── non_flooded/
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── test/
    ├── flooded/
    └── non_flooded/
```

## 🏗️ Architecture Details

### 1. Custom CNN (Trained from Scratch)

**Architecture**:
- 5 Convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Channel progression: 3 → 32 → 64 → 128 → 256 → 512
- Global Average Pooling
- 2 Dense layers with Dropout (0.5)
- Binary output with Sigmoid

**Parameters**: ~1.7M  
**Model Size**: 6.62 MB  
**Best For**: Edge deployment, resource-constrained environments

### 2. Pretrained Models

#### ResNet50
- **Base**: ImageNet pretrained
- **Parameters**: 24M total
- **Model Size**: 91.8 MB
- **Strategy**: Fine-tune last layers + custom head
- **Best For**: High accuracy requirements

#### EfficientNet-B0
- **Base**: ImageNet pretrained
- **Parameters**: 4.4M total
- **Model Size**: 16.66 MB
- **Strategy**: Fine-tune with efficient architecture
- **Best For**: Balanced accuracy and efficiency

#### MobileNetV2
- **Base**: ImageNet pretrained
- **Parameters**: 2.6M total
- **Model Size**: 9.86 MB
- **Strategy**: Lightweight mobile-first design
- **Best For**: Mobile/edge deployment

## 🚀 Quick Start

### Step 1: Setup Environment

```bash
# Activate virtual environment
source wldvenv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Step 2: Organize Data

Place your images in ImageFolder format:
```bash
data/train/flooded/
data/train/non_flooded/
```

### Step 3: Train Custom CNN

```bash
python train_flood_classifier.py \
  --model-type custom \
  --data-dir data/train \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --save-dir models/custom_cnn
```

### Step 4: Train Pretrained Model

```bash
# ResNet50
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name resnet50 \
  --data-dir data/train \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0001 \
  --save-dir models/resnet50

# EfficientNet-B0
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name efficientnet_b0 \
  --data-dir data/train \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0001 \
  --save-dir models/efficientnet_b0

# MobileNetV2
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name mobilenet_v2 \
  --data-dir data/train \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0001 \
  --save-dir models/mobilenet_v2
```

### Step 5: Evaluate Models

See `evaluate_flood_models.py` for comprehensive evaluation.

## 📈 Training Configuration

### Hyperparameters

| Parameter | Custom CNN | Pretrained Models |
|-----------|------------|-------------------|
| **Epochs** | 50 | 30 |
| **Batch Size** | 32 | 32 |
| **Learning Rate** | 0.001 | 0.0001 |
| **Optimizer** | Adam | Adam |
| **Loss** | Binary Cross-Entropy | Binary Cross-Entropy |
| **LR Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau |
| **Early Stopping** | 10 epochs patience | 10 epochs patience |

### Data Augmentation (Training)

- Random horizontal flip (50%)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop position variation (±5%)
- ImageNet normalization

### Preprocessing Pipeline

```python
1. Resize: shorter side → 488px (maintains aspect ratio)
2. LowerCrop: 244×244 at 85% from top (bottom middle)
3. Augmentation: flip + color jitter (training only)
4. ToTensor: PIL Image → PyTorch tensor
5. Normalize: ImageNet mean/std
```

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True Positives / (TP + FP)
- **Recall (Sensitivity)**: True Positives / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Area under ROC curve
- **Specificity**: True Negatives / (TN + FP)

### Efficiency Metrics
- **Inference Time**: Mean time per image (ms)
- **Model Size**: Storage in MB
- **FPS**: Frames per second capability

### Confusion Matrix
```
                Predicted
              Non-Flood  Flooded
Actual
Non-Flood        TN        FP
Flooded          FN        TP
```

## 🔍 Model Interpretability

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions the model focuses on:

```python
from src.gradcam_visualizer import FloodGradCAMVisualizer

# Create visualizer
visualizer = FloodGradCAMVisualizer(model, device)

# Visualize predictions
visualizer.visualize_batch(test_loader, num_samples=10)
```

**Expected Output**:
- Original image
- Heatmap showing attention regions
- Overlay with prediction confidence
- Should focus on road/water areas (bottom portion)

## 📁 Project Structure

```
windsurf-project/
├── src/
│   ├── flood_classifier.py         # Model architectures
│   ├── flood_preprocessing.py      # Custom preprocessing pipeline
│   ├── flood_evaluator.py          # Evaluation framework
│   ├── gradcam_visualizer.py       # Grad-CAM implementation
│   └── model_comparator.py         # Model comparison tools
│
├── examples/
│   ├── flood_data_pipeline_example.py
│   ├── test_lower_crop.py
│   └── README_PREPROCESSING.md
│
├── train_flood_classifier.py       # Training script
├── evaluate_flood_models.py        # Evaluation script (to create)
│
├── models/                          # Trained models
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── efficientnet_b0/
│   └── mobilenet_v2/
│
├── output/                          # Results
│   ├── evaluations/
│   ├── gradcam/
│   └── comparisons/
│
└── reports/                         # Generated reports
    └── model_comparison_report.md
```

## 📝 Expected Results (Benchmarks)

### Target Metrics

| Metric | Custom CNN | ResNet50 | EfficientNet-B0 | MobileNetV2 |
|--------|------------|----------|-----------------|-------------|
| **Accuracy** | >90% | >95% | >93% | >92% |
| **F1-Score** | >0.88 | >0.94 | >0.91 | >0.90 |
| **ROC-AUC** | >0.92 | >0.97 | >0.95 | >0.94 |
| **Inference (ms)** | <5 | <15 | <8 | <6 |
| **Model Size (MB)** | 6.6 | 91.8 | 16.7 | 9.9 |

## 🎓 Usage Examples

### Example 1: Train Custom CNN

```bash
python train_flood_classifier.py \
  --model-type custom \
  --data-dir data/train \
  --epochs 50 \
  --batch-size 32 \
  --save-dir models/custom_cnn
```

### Example 2: Fine-tune ResNet50

```bash
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name resnet50 \
  --data-dir data/train \
  --epochs 30 \
  --lr 0.0001 \
  --save-dir models/resnet50
```

### Example 3: Evaluate All Models

```python
from src.flood_evaluator import FloodDetectionEvaluator
from src.flood_classifier import create_model
import torch

# Load model
checkpoint = torch.load('models/custom_cnn/best_model.pth')
model = create_model('custom')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
evaluator = FloodDetectionEvaluator(model, device)
results = evaluator.evaluate(test_loader)

# Visualize
evaluator.plot_confusion_matrix(save_path='output/cm.png')
evaluator.plot_roc_curve(save_path='output/roc.png')
```

### Example 4: Compare Models

```python
from src.model_comparator import ModelComparator

comparator = ModelComparator()
comparator.add_model('Custom CNN', 'output/custom_results.json', 'custom', 6.62)
comparator.add_model('ResNet50', 'output/resnet50_results.json', 'pretrained', 91.8)
comparator.add_model('EfficientNet-B0', 'output/effnet_results.json', 'pretrained', 16.7)

# Generate comparison
comparator.print_summary()
comparator.plot_metrics_comparison(save_path='output/comparison.png')
comparator.generate_report(save_path='reports/comparison.md')
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
python train_flood_classifier.py --batch-size 16

# Or use CPU
python train_flood_classifier.py --device cpu
```

### Issue: Poor Convergence

**Solution**:
- Reduce learning rate: `--lr 0.0001`
- Increase epochs: `--epochs 100`
- Check data augmentation is not too aggressive

### Issue: Overfitting

**Solution**:
- Increase dropout in custom CNN
- Use more data augmentation
- Enable early stopping (already default)

### Issue: Low Accuracy

**Checklist**:
- ✓ Images properly labeled in correct folders
- ✓ Class balance (equal flooded/non-flooded)
- ✓ Image quality sufficient
- ✓ Preprocessing matches training (244×244, bottom crop)

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `src/flood_classifier.py` | Model definitions |
| `src/flood_preprocessing.py` | Custom LowerCrop transform |
| `src/flood_evaluator.py` | Metrics and evaluation |
| `src/gradcam_visualizer.py` | Model interpretability |
| `src/model_comparator.py` | Multi-model comparison |
| `train_flood_classifier.py` | Training script |

## 🎯 Next Steps

1. ✅ **Collect Data**: Organize 1,200 images in ImageFolder format
2. ✅ **Train Models**: Run training for all architectures
3. ✅ **Evaluate**: Generate metrics and visualizations
4. ✅ **Compare**: Use ModelComparator for analysis
5. ✅ **Deploy**: Select best model based on requirements

## 📖 Additional Resources

- **Preprocessing Guide**: `examples/README_PREPROCESSING.md`
- **Dataset Collection**: `DATASET_COLLECTION_GUIDE.md`
- **Training Requirements**: `MODEL_TRAINING_REQUIREMENTS.md`
- **Quick Start**: `QUICKSTART.md`

## 🏆 Success Criteria

- [x] Custom CNN implemented (5 conv layers)
- [x] 3 Pretrained models integrated
- [x] Training with early stopping
- [x] Comprehensive evaluation metrics
- [x] Grad-CAM visualization
- [x] Model comparison framework
- [ ] Achieve >90% accuracy on test set
- [ ] Generate comparison report
- [ ] Deploy best model

---

**Project Status**: ✅ **Code Complete - Ready for Training**

Once you have your dataset organized, you can start training immediately using the provided scripts!
