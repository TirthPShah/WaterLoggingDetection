# Flood Classification - Quick Start Guide

## 🚀 Complete Project Setup in 5 Steps

### Step 1: Organize Your Dataset (5 minutes)

```bash
# Required structure:
data/
├── train/
│   ├── flooded/       # 600 images
│   └── non_flooded/   # 600 images
└── test/
    ├── flooded/       # Test images
    └── non_flooded/   # Test images
```

### Step 2: Train Custom CNN (2-4 hours)

```bash
source wldvenv/bin/activate

python train_flood_classifier.py \
  --model-type custom \
  --data-dir data/train \
  --epochs 50 \
  --batch-size 32 \
  --save-dir models/custom_cnn
```

**Expected Output**: ~90-92% validation accuracy

### Step 3: Train Pretrained Models (1-2 hours each)

```bash
# ResNet50 (Best accuracy)
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name resnet50 \
  --data-dir data/train \
  --epochs 30 \
  --lr 0.0001 \
  --save-dir models/resnet50

# EfficientNet-B0 (Balanced)
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name efficientnet_b0 \
  --data-dir data/train \
  --epochs 30 \
  --lr 0.0001 \
  --save-dir models/efficientnet_b0

# MobileNetV2 (Fastest)
python train_flood_classifier.py \
  --model-type pretrained \
  --model-name mobilenet_v2 \
  --data-dir data/train \
  --epochs 30 \
  --lr 0.0001 \
  --save-dir models/mobilenet_v2
```

### Step 4: Evaluate All Models (30 minutes)

```bash
python evaluate_flood_models.py \
  --test-dir data/test \
  --models-dir models \
  --output-dir output/evaluations \
  --compare
```

**Generates**:
- Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrices
- ROC curves
- Grad-CAM visualizations
- Comparison report

### Step 5: Review Results

```bash
# View comparison report
cat output/evaluations/comparison/comparison_report.md

# Check visualizations
open output/evaluations/comparison/metrics_comparison.png
open output/evaluations/*/gradcam/
```

---

## 📊 What Gets Created

### After Training:

```
models/
├── custom_cnn/
│   ├── best_model.pth           # Best validation accuracy
│   ├── final_model.pth          # Final epoch
│   └── training_history.json    # Loss/accuracy curves
├── resnet50/
│   └── ...
├── efficientnet_b0/
│   └── ...
└── mobilenet_v2/
    └── ...
```

### After Evaluation:

```
output/evaluations/
├── custom_cnn/
│   ├── evaluation_results.json  # All metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.png
│   └── gradcam/                 # Visual explanations
│       ├── gradcam_sample_1.png
│       └── ...
├── resnet50/
│   └── ...
├── efficientnet_b0/
│   └── ...
├── mobilenet_v2/
│   └── ...
└── comparison/
    ├── comparison_report.md      # Comprehensive report
    ├── metrics_comparison.png    # Bar chart
    ├── efficiency_comparison.png # Speed/size
    └── accuracy_vs_efficiency.png # Scatter plot
```

---

## 🎯 Expected Performance

| Model | Accuracy | F1-Score | Inference | Size |
|-------|----------|----------|-----------|------|
| Custom CNN | ~91% | ~0.89 | 3-5 ms | 6.6 MB |
| ResNet50 | ~95%+ | ~0.94+ | 10-15 ms | 91.8 MB |
| EfficientNet-B0 | ~93% | ~0.91 | 5-8 ms | 16.7 MB |
| MobileNetV2 | ~92% | ~0.90 | 4-6 ms | 9.9 MB |

---

## ⚡ Training Tips

### For Best Accuracy
- Use ResNet50 or EfficientNet-B0
- Train for 30-50 epochs
- Use learning rate 0.0001
- Enable all data augmentation

### For Speed
- Use MobileNetV2 or Custom CNN
- Smaller batch sizes on CPU
- Reduce image size if needed

### If Overfitting
- Increase dropout
- More data augmentation
- Reduce model complexity
- Early stopping (enabled by default)

### If Underfitting
- Train longer (more epochs)
- Reduce dropout
- Use pretrained model
- Check data quality

---

## 🔍 Understanding Results

### Confusion Matrix
```
                Predicted
              Non-Flood  Flooded
Actual
Non-Flood       120        5      ← Good specificity
Flooded          3       122      ← Good sensitivity
```

### Grad-CAM Heatmap
- **Red/Yellow**: High attention (model focuses here)
- **Blue/Green**: Low attention
- **Expected**: Focus on road/water area (bottom of image)

### ROC Curve
- **AUC = 1.0**: Perfect classifier
- **AUC > 0.95**: Excellent
- **AUC > 0.90**: Good
- **AUC < 0.80**: Needs improvement

---

## 🛠️ Common Commands

### Check Training Progress
```bash
# View training history
python -c "
import json
with open('models/custom_cnn/training_history.json') as f:
    h = json.load(f)
    print(f'Best Val Acc: {h[\"best_val_acc\"]:.2f}%')
    print(f'Final Val Acc: {h[\"val_accs\"][-1]:.2f}%')
"
```

### Test Single Image
```python
import torch
from PIL import Image
from src.flood_classifier import create_model
from src.flood_preprocessing import get_flood_detection_transforms

# Load model
model = create_model('custom')
checkpoint = torch.load('models/custom_cnn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
transform = get_flood_detection_transforms(train=False)
image = Image.open('test_image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prob = output.item()
    pred = 'Flooded' if prob > 0.5 else 'Non-Flooded'
    
print(f'Prediction: {pred} ({prob:.2%} confidence)')
```

### Resume Training
```bash
# Load from checkpoint and continue
python train_flood_classifier.py \
  --model-type custom \
  --data-dir data/train \
  --epochs 100 \
  --resume models/custom_cnn/final_model.pth
```

---

## 📈 Monitoring Training

Training outputs real-time progress:
```
Epoch 10/50
----------------------------------------------------------------------
Training: 100%|████████| 30/30 [00:45<00:00, loss: 0.234, acc: 92.5%]
Validation: 100%|██████| 8/8 [00:08<00:00, loss: 0.189, acc: 94.2%]

Epoch 10 Summary:
  Train Loss: 0.234000 | Train Acc: 92.50%
  Val Loss: 0.189000 | Val Acc: 94.20%
  Learning Rate: 1.00e-03
  Time: 53.2s
  ✅ Saved best model (Val Acc: 94.20%)
```

---

## 🎓 Key Files Reference

| File | Purpose |
|------|---------|
| `train_flood_classifier.py` | Train models |
| `evaluate_flood_models.py` | Evaluate and compare |
| `src/flood_classifier.py` | Model architectures |
| `src/flood_preprocessing.py` | Data preprocessing |
| `src/flood_evaluator.py` | Metrics calculation |
| `src/gradcam_visualizer.py` | Visual explanations |
| `src/model_comparator.py` | Model comparison |

---

## ✅ Success Checklist

- [ ] Dataset organized (1200 images: 600 flooded, 600 non-flooded)
- [ ] Custom CNN trained (>90% accuracy)
- [ ] At least one pretrained model trained (>92% accuracy)
- [ ] Evaluation completed with all metrics
- [ ] Grad-CAM visualizations generated
- [ ] Comparison report reviewed
- [ ] Best model selected for deployment

---

## 🚨 Troubleshooting

**Error: "RuntimeError: CUDA out of memory"**
```bash
# Solution 1: Reduce batch size
--batch-size 16

# Solution 2: Use CPU
--device cpu
```

**Error: "FileNotFoundError: data/train"**
```bash
# Check directory structure
ls -R data/train
# Should show: flooded/ and non_flooded/ subdirectories
```

**Low Accuracy (<80%)**
- Check image labels are correct
- Verify equal class distribution
- Ensure images are clear quality
- Try pretrained model instead

**Training Too Slow**
- Reduce `--epochs`
- Increase `--batch-size` (if GPU memory allows)
- Use `--num-workers 8` for faster data loading
- Consider using GPU if on CPU

---

## 📞 Need Help?

See detailed guides:
- `FLOOD_DETECTION_PROJECT_GUIDE.md` - Complete project guide
- `examples/README_PREPROCESSING.md` - Preprocessing details
- `DATASET_COLLECTION_GUIDE.md` - Dataset information

---

**Ready to start!** Just organize your data and run the training commands above.
