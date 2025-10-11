# 🚀 Quick Start Guide

Get up and running with the AI-CCTV Waterlogging Detection system in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone and enter directory
cd windsurf-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Create Sample Data (30 seconds)

```bash
# Generate sample weather data
python demo.py --create-sample-data
```

This creates `data/sample_weather.csv` with synthetic weather data for testing.

## Step 3: Test with Sample Video (2 minutes)

### Option A: Use your own video
```bash
python demo.py --video /path/to/your/video.mp4
```

### Option B: Use sample images
```bash
# Create a directory with your CCTV images
mkdir -p data/sample_images
# Copy your images to data/sample_images/

# Run demo
python demo.py --video data/sample_images/
```

## Step 4: View Results

After processing completes, check the `output/` directory:

```
output/
├── output_video.mp4          # Processed video with risk overlays
├── frames/                    # Individual processed frames
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── results_TIMESTAMP.json     # Detailed results
├── summary_TIMESTAMP.json     # Summary statistics
└── temporal_plot.png          # Risk over time graph
```

## Step 5: Understand the Output

### Risk Visualization
- **Green overlay**: Low risk (0-30%)
- **Orange overlay**: Medium risk (30-60%)
- **Red overlay**: High risk (60-100%)

### Results JSON
```json
{
  "fused_risk": {
    "mean": 0.15,    // Average risk across all frames
    "max": 0.45,     // Maximum risk detected
    "min": 0.05      // Minimum risk detected
  },
  "risk_levels": {
    "low": 70,       // Number of frames with low risk
    "medium": 25,    // Number of frames with medium risk
    "high": 5        // Number of frames with high risk
  }
}
```

## 🎯 Next Steps

### Train Your Own Detection Model

If you have labeled data (images + segmentation masks):

```bash
python train_detection.py \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --epochs 50
```

### Use Custom Weather Data

Prepare a CSV file with this format:
```csv
timestamp,rainfall_mm,humidity_percent,temperature_c
2024-01-01T00:00:00,0.5,75.0,25.0
2024-01-01T00:10:00,1.2,78.0,24.5
```

Then run:
```bash
python demo.py --video video.mp4 --weather your_weather.csv
```

### Load Pretrained Models

```bash
python demo.py \
  --video video.mp4 \
  --detection-model models/detection/best_model.pth \
  --forecast-model models/forecasting/lstm_forecaster.pth
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Change detection model
DETECTION_MODEL_NAME = "deeplabv3+"  # Better accuracy
ENCODER_NAME = "resnet50"            # Stronger backbone

# Adjust sensitivity
DETECTION_THRESHOLD = 0.3  # Lower = more sensitive

# Change fusion weights
DETECTION_WEIGHT = 0.7  # More weight on current detection
FORECAST_WEIGHT = 0.3   # Less weight on forecast
```

## 📊 Performance Tips

### Speed Up Processing
```python
# In config.py
IMAGE_SIZE = (256, 256)      # Smaller images
FRAME_SKIP = 10              # Process fewer frames
ENCODER_NAME = "mobilenet_v2"  # Faster encoder
```

### Improve Accuracy
```python
# In config.py
IMAGE_SIZE = (512, 512)      # Larger images
DETECTION_MODEL_NAME = "deeplabv3+"  # Better model
ENCODER_NAME = "resnet101"   # Stronger encoder
```

### Reduce False Positives
```python
# In config.py
DETECTION_THRESHOLD = 0.6    # Higher threshold
MIN_WATERLOG_AREA = 500      # Larger minimum area
MORPH_KERNEL_SIZE = 7        # More aggressive filtering
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# In config.py
IMAGE_SIZE = (256, 256)  # Reduce image size
# Or process on CPU:
DEVICE = "cpu"
```

### Video Won't Process
```bash
# Check video codec
ffmpeg -i your_video.mp4

# Convert to compatible format
ffmpeg -i input.mov -c:v libx264 output.mp4
```

### Poor Detection Quality
1. Ensure good lighting in CCTV footage
2. Train model on similar camera angles/conditions
3. Adjust detection threshold
4. Use better weather data

## 📚 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **API Reference**: Check docstrings in source code
- **Training Guide**: See [train_detection.py](train_detection.py)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## 💡 Common Use Cases

### 1. Real-time Monitoring
```python
from src.pipeline import WaterloggingPipeline
import config

pipeline = WaterloggingPipeline(config)

# Process live stream
for frame in camera_stream:
    vis_frame, results = pipeline.process_single_frame(
        frame, metadata, weather_data
    )
    display(vis_frame)
```

### 2. Batch Processing
```bash
# Process multiple videos
for video in videos/*.mp4; do
    python demo.py --video "$video" --output "results/$(basename $video)"
done
```

### 3. Alert System
```python
# In your code
if results['fusion_metadata']['risk_level'] == 'high':
    send_alert(f"High waterlogging risk: {fused_risk:.1%}")
```

## ✅ Verification

To verify everything is working:

```bash
# 1. Create sample data
python demo.py --create-sample-data

# 2. Should see: "Sample weather data created at: data/sample_weather.csv"

# 3. Test with sample (if you have images/video)
python demo.py --video your_test_video.mp4

# 4. Check output directory
ls -la output/
```

Expected output structure:
```
output/
├── output_video.mp4
├── frames/
├── results_*.json
├── summary_*.json
└── temporal_plot.png

logs/
└── system_*.log
```

---

**Ready to go! 🎉** If you encounter issues, check [README.md](README.md) or open an issue on GitHub.
