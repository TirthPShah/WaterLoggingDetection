# Flood Detection Model Comparison Report

## Executive Summary

This report compares the performance of multiple flood detection models:
- **Custom CNN** (trained from scratch)
- **Pretrained Models** (ResNet50, EfficientNet-B0, MobileNetV2)

### Key Findings

- **Best Accuracy**: ResNet50 (0.9160)
- **Best F1-Score**: ResNet50 (0.9179)
- **Fastest Inference**: Custom CNN (14.00 ms)
- **Smallest Model**: Custom CNN (6.63 MB)

---

## Performance Comparison Table

| Model | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference (ms) | Size (MB) |
|-------|------|----------|-----------|--------|----------|---------|----------------|-----------|
| Custom CNN | custom | 0.7901 | 0.7439 | 0.9037 | 0.8161 | 0.8802 | 14.00 | 6.63 |
| ResNet50 | pretrained | 0.9160 | 0.9248 | 0.9111 | 0.9179 | 0.9726 | 94.37 | 92.01 |

---

## Detailed Analysis

### 1. Accuracy Metrics


#### Custom CNN
- **Accuracy**: 0.7901 (79.01%)
- **Precision**: 0.7439
- **Recall**: 0.9037
- **F1-Score**: 0.8161
- **ROC-AUC**: 0.8802

#### ResNet50
- **Accuracy**: 0.9160 (91.60%)
- **Precision**: 0.9248
- **Recall**: 0.9111
- **F1-Score**: 0.9179
- **ROC-AUC**: 0.9726

### 2. Efficiency Metrics


#### Custom CNN
- **Inference Time**: 14.00 ms/image
- **Model Size**: 6.63 MB
- **FPS**: 71.45

#### ResNet50
- **Inference Time**: 94.37 ms/image
- **Model Size**: 92.01 MB
- **FPS**: 10.60

---

## Recommendations

### For Production Deployment

### For Edge Devices

**Recommended**: ResNet50
- Good accuracy with small size and fast inference

---

## Conclusion

This comprehensive comparison demonstrates the trade-offs between model architectures.
Custom CNNs offer good performance with smaller size, while pretrained models leverage
transfer learning for potentially better accuracy at the cost of larger model size.

