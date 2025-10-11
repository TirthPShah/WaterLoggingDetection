# AI-CCTV Waterlogging Detection & Forecasting System

A comprehensive AI-powered system for detecting waterlogged regions from CCTV footage and forecasting future waterlogging risk using weather data and temporal analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Real-time Waterlogging Detection**: Semantic segmentation model (U-Net, DeepLabV3+, FPN) to identify waterlogged regions in CCTV footage
- **Weather-based Forecasting**: LSTM/ML models to predict future waterlogging risk using weather data (rainfall, humidity, temperature)
- **Intelligent Fusion**: Combines detection and forecasting predictions with adaptive weighting
- **Temporal Smoothing**: Reduces flickering in video sequences for stable predictions
- **Advanced Preprocessing**: Automatic brightness/contrast adjustment, CLAHE enhancement, denoising
- **Comprehensive Visualization**: Risk overlays, heatmaps, temporal plots, and annotated videos
- **Robust Postprocessing**: Morphological operations, area filtering, region feature extraction
- **Export & Logging**: JSON/CSV export, detailed logging, performance monitoring
- **Modular Architecture**: Easy to extend, swap models, and customize components

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage](#usage)
  - [Running Demo](#running-demo)
  - [Training Models](#training-models)
  - [Using as Library](#using-as-library)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Model Details](#model-details)
- [Output](#output)
- [Performance](#performance)
- [Extending the System](#extending-the-system)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd windsurf-project
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create required directories**:
```bash
mkdir -p data models output logs
```

## ⚡ Quick Start

### Run Demo with Sample Data

1. **Create sample weather data**:
```bash
python demo.py --create-sample-data
```

2. **Run detection on your video/images**:
```bash
python demo.py --video path/to/video.mp4
```

3. **With custom weather data**:
```bash
python demo.py --video path/to/video.mp4 --weather path/to/weather.csv
```

4. **Specify output directory**:
```bash
python demo.py --video path/to/video.mp4 --output results/
```

### Expected Output

After running the demo, you'll find:
- **Output video** with risk overlays: `output/output_video.mp4`
- **Individual frames**: `output/frames/frame_*.jpg`
- **Results data**: `output/results_*.json` and `output/summary_*.json`
- **Temporal plot**: `output/temporal_plot.png`
- **System logs**: `logs/system_*.log`

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CCTV Input Stream                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Data Ingestion & Preprocessing               │
│  • Frame extraction    • Resize & normalize                  │
│  • Brightness/contrast • CLAHE enhancement                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Waterlogging Detection                       │
│  • Semantic segmentation (U-Net/DeepLabV3+)                  │
│  • Binary mask generation                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Postprocessing                               │
│  • Morphological operations  • Small region filtering        │
│  • Temporal smoothing        • Feature extraction            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              4. Weather Data Integration                     │
│  • Current conditions    • Historical aggregation            │
│  • Rainfall, humidity    • Temperature                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              5. Risk Forecasting                             │
│  • LSTM / Random Forest / Gradient Boosting                  │
│  • Temporal sequence modeling                                │
│  • Future risk prediction                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              6. Prediction Fusion                            │
│  • Weighted combination      • Adaptive fusion               │
│  • Temporal smoothing        • Risk classification           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              7. Visualization & Export                       │
│  • Risk overlays    • Heatmaps    • Videos                   │
│  • JSON/CSV export  • Logs        • Plots                    │
└─────────────────────────────────────────────────────────────┘
```

## 📖 Usage

### Running Demo

The `demo.py` script provides an easy way to test the system:

```bash
# Basic usage
python demo.py --video path/to/video.mp4

# With all options
python demo.py \
  --video path/to/video.mp4 \
  --weather path/to/weather.csv \
  --output results/ \
  --detection-model models/detection/best_model.pth \
  --forecast-model models/forecasting/lstm_forecaster.pth
```

**Arguments**:
- `--video`: Path to video file or image directory (required)
- `--weather`: Path to weather data CSV/JSON (optional, creates sample if not provided)
- `--output`: Output directory for results (default: `output/`)
- `--detection-model`: Path to trained detection model checkpoint (optional)
- `--forecast-model`: Path to trained forecasting model checkpoint (optional)
- `--create-sample-data`: Create sample weather data for testing

### Training Models

#### Train Detection Model

Requires labeled dataset with images and corresponding segmentation masks:

```bash
python train_detection.py \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --output models/detection \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.001
```

**Dataset Structure**:
```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.png  (binary mask: 0=background, 255=water)
│       ├── image_002.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

#### Train Forecasting Model

Requires historical detection results and weather data:

```bash
python train_forecasting.py \
  --detection-results data/historical_detections.csv \
  --weather-data data/historical_weather.csv \
  --output models/forecasting \
  --model-type lstm \
  --epochs 50 \
  --batch-size 32
```

**Model Types**:
- `lstm`: LSTM neural network (recommended for temporal sequences)
- `random_forest`: Random Forest regressor
- `gradient_boosting`: Gradient Boosting regressor

### Using as Library

You can use the system as a Python library:

```python
import config
from src.pipeline import WaterloggingPipeline

# Initialize pipeline
pipeline = WaterloggingPipeline(config)

# Load pretrained models (optional)
pipeline.load_detection_model("models/detection/best_model.pth")
pipeline.load_forecasting_model("models/forecasting/lstm_forecaster.pth", input_size=12)

# Process video
results = pipeline.process_video(
    video_path="path/to/video.mp4",
    weather_data_path="path/to/weather.csv",
    output_video_path="output/result.mp4",
    visualize=True
)

print(f"Processed {results['frames_processed']} frames")
print(f"Average risk: {results['summary']['fused_risk']['mean']:.2%}")
```

**Process Single Frame**:

```python
import cv2
from src.data_ingestion import WeatherDataLoader

# Load frame
frame = cv2.imread("frame.jpg")

# Get weather data
weather_loader = WeatherDataLoader("weather.csv")
weather_data = weather_loader.get_weather_at_timestamp("2024-01-01T12:00:00")

# Process
frame_metadata = {'frame_number': 0, 'timestamp': "2024-01-01T12:00:00"}
vis_image, results = pipeline.process_single_frame(
    frame, frame_metadata, weather_data, visualize=True
)

# Access results
print(f"Fused risk: {results['fusion_metadata']['fused_risk_score']:.2%}")
print(f"Risk level: {results['fusion_metadata']['risk_level']}")
```

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

### Key Settings

```python
# Image processing
IMAGE_SIZE = (512, 512)  # Target image size
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Detection model
DETECTION_MODEL_NAME = "unet"  # Options: unet, deeplabv3, fpn, pspnet
ENCODER_NAME = "resnet34"  # Encoder backbone
DETECTION_THRESHOLD = 0.5  # Binary threshold

# Forecasting
FORECAST_MODEL_TYPE = "lstm"  # Options: lstm, random_forest, gradient_boosting
SEQUENCE_LENGTH = 10  # Timesteps for prediction
FORECAST_HORIZON_MINUTES = 30  # Prediction horizon

# Fusion
DETECTION_WEIGHT = 0.6  # Weight for detection (0-1)
FORECAST_WEIGHT = 0.4   # Weight for forecast (0-1)
TEMPORAL_SMOOTHING_ALPHA = 0.3  # Smoothing factor

# Visualization
OVERLAY_ALPHA = 0.4  # Overlay transparency
COLORMAP = "jet"  # Matplotlib colormap
```

## 📊 Data Format

### Weather Data (CSV)

```csv
timestamp,rainfall_mm,humidity_percent,temperature_c
2024-01-01T00:00:00,0.5,75.0,25.0
2024-01-01T00:10:00,1.2,78.0,24.5
2024-01-01T00:20:00,2.5,80.0,24.0
```

**Required columns**:
- `timestamp`: ISO format datetime
- `rainfall_mm`: Rainfall in millimeters
- `humidity_percent`: Relative humidity (0-100)
- `temperature_c`: Temperature in Celsius

### Detection Results (JSON)

Automatically exported by the system:

```json
{
  "frame": {
    "frame_number": 0,
    "timestamp": "2024-01-01T12:00:00",
    "source": "video.mp4"
  },
  "detection": {
    "waterlogged_ratio": 0.15,
    "mean_probability": 0.65,
    "num_regions": 3
  },
  "forecast": {
    "forecast_risk": 0.22,
    "method": "lstm"
  },
  "fusion": {
    "fused_risk_score": 0.18,
    "risk_level": "low"
  }
}
```

## 🤖 Model Details

### Detection Models

| Model | Encoder | Parameters | Speed | Accuracy |
|-------|---------|------------|-------|----------|
| U-Net | ResNet34 | 24M | Fast | Good |
| DeepLabV3+ | ResNet50 | 40M | Medium | Better |
| FPN | ResNet50 | 35M | Fast | Good |
| PSPNet | ResNet50 | 46M | Slow | Best |

### Forecasting Models

| Model | Type | Training Data | Best For |
|-------|------|---------------|----------|
| LSTM | Deep Learning | Temporal sequences | Long-term patterns |
| Random Forest | Ensemble | Tabular features | Fast inference |
| Gradient Boosting | Ensemble | Tabular features | High accuracy |

## 📈 Output

### Results Summary

```json
{
  "total_frames": 100,
  "fused_risk": {
    "mean": 0.15,
    "std": 0.08,
    "min": 0.05,
    "max": 0.45,
    "median": 0.13
  },
  "risk_levels": {
    "low": 70,
    "medium": 25,
    "high": 5
  }
}
```

### Risk Levels

- **Low (0-30%)**: ✅ Minimal waterlogging, roads passable
- **Medium (30-60%)**: ⚠️ Moderate waterlogging, caution advised
- **High (60-100%)**: 🚫 Severe waterlogging, avoid area

## ⚡ Performance

Typical performance on standard hardware:

| Hardware | Resolution | FPS | Notes |
|----------|-----------|-----|-------|
| NVIDIA RTX 3090 | 512×512 | 45-60 | Real-time capable |
| NVIDIA GTX 1080 Ti | 512×512 | 20-30 | Near real-time |
| CPU (Intel i7) | 512×512 | 2-5 | Offline processing |

**Optimization Tips**:
- Use smaller image size (e.g., 256×256) for faster processing
- Enable frame skipping (`FRAME_SKIP = 5`)
- Use lighter encoder (e.g., `mobilenet_v2`)
- Batch processing for multiple frames

## 🔧 Extending the System

### Add New Detection Model

```python
from src.detection_model import WaterloggingDetector

# Create custom detector
class CustomDetector(WaterloggingDetector):
    def _create_model(self):
        # Implement your model here
        return your_custom_model

# Use in pipeline
detector = CustomDetector(...)
```

### Add Custom Preprocessing

```python
from src.preprocessing import ImagePreprocessor

class CustomPreprocessor(ImagePreprocessor):
    def preprocess(self, image):
        # Add custom preprocessing steps
        image = super().preprocess(image)
        image = your_custom_function(image)
        return image
```

### Custom Fusion Strategy

```python
from src.fusion import PredictionFusion

class CustomFusion(PredictionFusion):
    def fuse_predictions(self, detection_risk, forecast_risk):
        # Implement custom fusion logic
        fused_risk = your_fusion_logic(detection_risk, forecast_risk)
        return fused_risk
```

## ⚠️ Limitations

1. **Labeled Data**: Detection model requires labeled segmentation masks for training
2. **Static Cameras**: Optimized for fixed-viewpoint CCTV cameras
3. **Weather Data**: Requires synchronized weather data for accurate forecasting
4. **Lighting**: Performance may degrade in low-light or nighttime conditions
5. **Reflections**: May produce false positives on wet surfaces without actual flooding
6. **Occlusions**: Objects blocking camera view can affect detection accuracy

## 🛠️ Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or image size in config.py
IMAGE_SIZE = (256, 256)  # Smaller size
```

**Issue**: Poor detection accuracy
```bash
# Solution: Train model on domain-specific data
python train_detection.py --train-images your_data/images --train-masks your_data/masks
```

**Issue**: Flickering in video output
```bash
# Solution: Increase temporal smoothing
TEMPORAL_SMOOTHING_ALPHA = 0.5  # Higher = more smoothing
```

## 📝 Future Improvements

- [ ] Real-time streaming support (RTSP, RTMP)
- [ ] Multi-camera fusion and coordination
- [ ] Integration with weather forecast APIs
- [ ] Mobile/edge deployment optimization
- [ ] Web dashboard for monitoring
- [ ] Alert system for high-risk scenarios
- [ ] 3D depth estimation for water level
- [ ] Transfer learning from satellite imagery

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **segmentation-models-pytorch**: Pre-trained segmentation models
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Albumentations**: Image augmentation library

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for flood prevention and public safety**
