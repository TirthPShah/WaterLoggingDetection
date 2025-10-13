# Model Training Requirements - AI-CCTV Waterlogging Detection System

## 📊 SECTION 1 — DETECTION MODEL REQUIREMENTS

### Current Status
- ✅ **Detection Models**: Fully coded (U-Net, DeepLabV3+, FPN, PSPNet)
- ❌ **Trained Weights**: Not available - requires training
- ❌ **Dataset**: Not collected yet

### Dataset Required

**Type**: Image or video frames with waterlogging regions labeled as segmentation masks

**Purpose**: Train the semantic segmentation model to identify flooded areas in CCTV footage

**Format**: 
- Images: JPG/PNG
- Masks: Binary masks (same resolution as images)

**Structure**:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Best Dataset Sources

#### Option A: AI4Floods Dataset (Recommended)
- **Source**: Google Research / Kaggle
- **URL**: https://www.kaggle.com/datasets/ratthachat/ai4floods
- **Description**: Drone and CCTV-style flood imagery with masks
- **Quality**: High-quality annotations, diverse scenarios

#### Option B: FloodNet Dataset
- **Source**: University at Buffalo, SUNY
- **URL**: https://github.com/BinaLab/FloodNet
- **Description**: Labeled flood and non-flood regions from aerial and street-level views
- **Quality**: Research-grade, peer-reviewed

#### Option C: CrisisVision (xView2)
- **Source**: xView2 Challenge
- **URL**: https://xview2.org/
- **Description**: Post-disaster imagery for transfer learning
- **Quality**: Large-scale, diverse disaster scenarios

#### Option D: Synthetic Dataset (Fallback)
- **Script**: `utils/create_sample_dataset.py`
- **Use Case**: Quick testing, prototyping
- **Command**: 
  ```bash
  python utils/create_sample_dataset.py --num-images 500
  ```

### Pretrained Model Bases

**Recommended Encoder Backbones**:
- EfficientNet-B3 (TIMM library)
- ResNet34/ResNet50 (segmentation-models-pytorch)
- MobileNetV2 (for edge deployment)

**Installation**:
```bash
pip install segmentation-models-pytorch timm
```

These backbones come pretrained on ImageNet and are automatically downloaded.

### Training Requirements

**Hardware**:
- GPU: NVIDIA GTX 1080 Ti or higher (CUDA 11.8+)
- RAM: 16GB minimum
- Storage: 20GB minimum

**Dataset Size**:
- Minimum: 1000+ labeled images
- Recommended: 5000+ labeled images

**Hyperparameters**:
```python
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
OPTIMIZER = "Adam"
MODEL_NAME = "deeplabv3+"
ENCODER = "resnet50"
```

**Training Command**:
```bash
python train_detection.py \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --epochs 50 \
  --batch-size 8
```

**Expected Output**:
- `models/detection/waterlogging_detection.pth`
- Training logs in `logs/training_detection_*.log`
- Metrics visualization in `output/evaluation/`

---

## 🌦️ SECTION 2 — FORECASTING MODEL REQUIREMENTS

### Current Status
- ✅ **Forecasting Models**: Fully coded (LSTM, Random Forest, Gradient Boosting)
- ❌ **Trained Weights**: Not available
- ❌ **Weather Dataset**: Not collected

### Dataset Required

**Type**: Tabular time series data combining historical waterlogging detections and weather readings

**Required Columns**:
```csv
timestamp, rainfall_mm, humidity_percent, temperature_c, wind_speed, pressure, detected_waterlogged_ratio, previous_risk_score
```

**Format**: CSV or JSON

**Temporal Structure**:
- Window: 6-hour intervals
- Sequence length: 10 timesteps
- Total features: 12 per timestep

### Best Weather Data Sources

#### Option A: OpenWeatherMap API (Recommended)
- **URL**: https://openweathermap.org/api
- **Features**: 
  - Free tier: 1-hourly data
  - Historical and real-time data
  - Temperature, rainfall, humidity, wind, pressure
- **Cost**: Free up to 1000 calls/day
- **API Key**: Required (free signup)

#### Option B: Indian Meteorological Department (IMD)
- **URL**: https://mausam.imd.gov.in/
- **Features**: 
  - Official Indian weather data
  - Historical archives
  - Localized to Indian cities
- **Cost**: Free

#### Option C: NASA POWER API
- **URL**: https://power.larc.nasa.gov/
- **Features**:
  - Global coverage
  - Historical data (1981-present)
  - High accuracy
- **Cost**: Free

#### Option D: ERA5 Reanalysis Dataset
- **URL**: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- **Features**:
  - Comprehensive global weather data
  - Hourly data since 1979
- **Cost**: Free (requires registration)

### Historical Waterlogging Data

**Option 1**: Generate from Detection Model
- Run trained detection model on archived CCTV footage
- Extract waterlogged ratios and metrics
- Align with weather data timestamps

**Option 2**: Public Datasets
- Disaster Response Data (Kaggle)
- Flood Damage Assessment (Copernicus)
- UN-SPIDER disaster datasets

### Training Requirements

**Input Features**: 12 features per timestep
- Weather: rainfall, humidity, temperature, wind_speed, pressure (5 features)
- Detection: waterlogged_ratio, max_probability, region_count, etc. (5 features)
- Historical: previous_risk_score, 6h_rainfall_sum (2 features)

**Model Architecture**:
- LSTM: 2 layers, 64 hidden units
- Input sequence: 10 timesteps
- Output: Risk score (0-1)

**Training Command**:
```bash
python train_forecasting.py \
  --train data/weather/train.csv \
  --val data/weather/val.csv \
  --epochs 100 \
  --batch-size 32
```

**Expected Output**:
- `models/forecasting/waterlogging_forecast.pth`
- Training metrics in `logs/training_forecasting_*.log`

---

## 📦 SECTION 3 — DATA COLLECTION AND PREPARATION PIPELINE

### Step-by-Step Data Collection Process

#### Step 1: Download Flood Detection Dataset
```bash
# Option A: Download FloodNet from GitHub
git clone https://github.com/BinaLab/FloodNet.git data/raw/floodnet

# Option B: Download AI4Floods from Kaggle (requires Kaggle API)
kaggle datasets download -d ratthachat/ai4floods -p data/raw/
unzip data/raw/ai4floods.zip -d data/raw/ai4floods
```

#### Step 2: Standardize Images
```bash
# Use preprocessing script to resize all images to 512x512
python utils/prepare_dataset.py \
  --input data/raw/floodnet \
  --output data/processed \
  --size 512 512
```

#### Step 3: Split Dataset
```python
# Create train/val/test splits (70/20/10)
python utils/split_dataset.py \
  --input data/processed \
  --output data/ \
  --split 0.7 0.2 0.1
```

#### Step 4: Verify Integration
```python
# Test data loading
python -c "from src.data_ingestion import CCTVDataLoader; 
loader = CCTVDataLoader('data/train'); 
print(f'Dataset size: {len(loader)}')"
```

#### Step 5: Collect Weather Data
```bash
# Download historical weather data
python utils/download_weather.py \
  --location "Mumbai,IN" \
  --start-date "2023-01-01" \
  --end-date "2024-01-01" \
  --api-key YOUR_OPENWEATHER_API_KEY
```

#### Step 6: Align Data
```python
# Align weather data with image timestamps
python utils/align_weather_data.py \
  --images data/train/images \
  --weather data/weather/raw_weather.csv \
  --output data/weather/aligned_weather.csv
```

#### Step 7: Final Data Structure
```
data/
├── train/
│   ├── images/          # Training images
│   └── masks/           # Training masks
├── val/
│   ├── images/          # Validation images
│   └── masks/           # Validation masks
├── test/
│   ├── images/          # Test images
│   └── masks/           # Test masks
└── weather/
    ├── train.csv        # Training weather data
    ├── val.csv          # Validation weather data
    └── test.csv         # Test weather data
```

---

## 🎯 SECTION 4 — MODEL TRAINING PIPELINE

### Detection Model Training

**Full Training Command**:
```bash
python train_detection.py \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --model deeplabv3+ \
  --encoder resnet50 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cuda \
  --save-dir models/detection
```

**Expected Output**:
- `models/detection/waterlogging_detection.pth`
- `models/detection/best_model.pth` (best validation performance)
- `logs/training_detection_TIMESTAMP.log`

### Forecasting Model Training

**Full Training Command**:
```bash
python train_forecasting.py \
  --train data/weather/train.csv \
  --val data/weather/val.csv \
  --model lstm \
  --hidden-size 64 \
  --num-layers 2 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda \
  --save-dir models/forecasting
```

**Expected Output**:
- `models/forecasting/waterlogging_forecast.pth`
- `models/forecasting/best_model.pth`
- `logs/training_forecasting_TIMESTAMP.log`

### Update Configuration

After training, update `config.py`:
```python
# Model paths
DETECTION_MODEL_PATH = "models/detection/best_model.pth"
FORECAST_MODEL_PATH = "models/forecasting/best_model.pth"

# Mark as trained
USE_PRETRAINED_DETECTION = True
USE_PRETRAINED_FORECAST = True
```

---

## 📊 SECTION 5 — EVALUATION REQUIREMENTS

### Detection Model Metrics

**Primary Metrics**:
- IoU (Intersection over Union)
- F1 Score
- Precision
- Recall
- Pixel Accuracy

**Evaluation Command**:
```bash
python evaluate_detection.py \
  --model models/detection/best_model.pth \
  --test-images data/test/images \
  --test-masks data/test/masks \
  --output output/evaluation/detection
```

**Expected Output**:
- `output/evaluation/detection/metrics.json`
- `output/evaluation/detection/confusion_matrix.png`
- `output/evaluation/detection/sample_predictions/`

### Forecasting Model Metrics

**Primary Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Accuracy (for risk level classification: low/medium/high)
- R² Score

**Evaluation Command**:
```bash
python evaluate_forecasting.py \
  --model models/forecasting/best_model.pth \
  --test-data data/weather/test.csv \
  --output output/evaluation/forecasting
```

**Expected Output**:
- `output/evaluation/forecasting/metrics.json`
- `output/evaluation/forecasting/predictions_vs_actual.png`
- `output/evaluation/forecasting/risk_distribution.png`

---

## 🚀 SECTION 6 — OPTIONAL DEPLOYMENT FEATURES

### Real-time CCTV Integration
```python
# Using OpenCV for RTSP streams
from src.data_ingestion import RTSPStreamLoader

stream = RTSPStreamLoader("rtsp://camera_ip:port/stream")
for frame in stream:
    result = pipeline.process_single_frame(frame)
```

### REST API Endpoint
```bash
# Using FastAPI
pip install fastapi uvicorn

# Run API server
python api_server.py --host 0.0.0.0 --port 8000
```

**API Endpoints**:
- `POST /predict` - Upload frame and get prediction
- `POST /predict_video` - Upload video and get analysis
- `GET /health` - Health check
- `GET /models/info` - Model information

### Weather API Integration
```python
# Automatic weather fetching
from src.weather_api import OpenWeatherMapClient

weather_client = OpenWeatherMapClient(api_key="YOUR_KEY")
current_weather = weather_client.get_current("Mumbai,IN")
```

### Web Dashboard
```bash
# Using Streamlit
pip install streamlit

# Run dashboard
streamlit run dashboard.py
```

**Features**:
- Live video feed with predictions
- Risk level indicators
- Historical trends
- Alert notifications

### Alerting System
```python
# Telegram alerts
pip install python-telegram-bot

# Twilio SMS alerts
pip install twilio
```

---

## 💻 SECTION 7 — ENVIRONMENT SETUP

### Required Libraries
All listed in `requirements.txt`:
```
torch>=2.2.0
torchvision>=0.17.0
segmentation-models-pytorch>=0.3.3
timm>=0.9.0
opencv-python>=4.8.0
albumentations>=1.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0  # For weather API
```

### Hardware Recommendations

**For Training**:
- GPU: NVIDIA RTX 3090 / A100 (recommended)
- GPU: NVIDIA GTX 1080 Ti (minimum)
- CPU: Intel i7-10700K or AMD Ryzen 7 5800X
- RAM: 32GB (recommended), 16GB (minimum)
- Storage: 100GB SSD

**For Inference**:
- GPU: NVIDIA GTX 1660 or higher
- CPU: Intel i7 or higher (for CPU-only inference)
- RAM: 16GB
- Storage: 20GB

### Installation
```bash
# Create environment
python -m venv wldvenv
source wldvenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 🤖 SECTION 8 — RESOURCE COLLECTION CHECKLIST

### AI Agent Action Items

- [ ] **Task 1**: Download FloodNet or AI4Floods dataset
  - Source: GitHub/Kaggle
  - Format: Images + segmentation masks
  - Size: 5000+ images recommended

- [ ] **Task 2**: Download weather datasets
  - Source: OpenWeatherMap API
  - Time range: Multiple seasons, weather conditions
  - Format: CSV with hourly data

- [ ] **Task 3**: Preprocess datasets
  - Resize images to 512×512
  - Normalize masks to binary (0/1)
  - Split train/val/test (70/20/10)

- [ ] **Task 4**: Prepare pretrained encoders
  - Install segmentation-models-pytorch
  - Download ResNet50 ImageNet weights
  - Verify model instantiation

- [ ] **Task 5**: Train detection model
  - Run train_detection.py
  - Monitor training metrics
  - Save best checkpoint

- [ ] **Task 6**: Train forecasting model
  - Align weather + detection data
  - Run train_forecasting.py
  - Save best checkpoint

- [ ] **Task 7**: Evaluate models
  - Run evaluation scripts
  - Generate metrics reports
  - Create visualization samples

- [ ] **Task 8**: Update configuration
  - Set model paths in config.py
  - Update default parameters
  - Document model versions

- [ ] **Task 9**: Prepare demo data
  - Create sample video with predictions
  - Generate documentation
  - Package for deployment

---

## 📤 SECTION 9 — EXPECTED OUTPUT ARTIFACTS

### Model Files
```
models/
├── detection/
│   ├── waterlogging_detection.pth        # Final trained model
│   ├── best_model.pth                     # Best validation model
│   └── checkpoint_epoch_*.pth             # Intermediate checkpoints
└── forecasting/
    ├── waterlogging_forecast.pth          # Final trained model
    ├── best_model.pth                     # Best validation model
    └── checkpoint_epoch_*.pth             # Intermediate checkpoints
```

### Data Files
```
data/
├── train/                                 # Training data
├── val/                                   # Validation data
├── test/                                  # Test data
└── weather/
    ├── weather.csv                        # Complete weather data
    ├── train.csv                          # Training weather
    ├── val.csv                            # Validation weather
    └── test.csv                           # Test weather
```

### Evaluation Outputs
```
output/evaluation/
├── detection/
│   ├── metrics.json                       # Detection metrics
│   ├── confusion_matrix.png               # Confusion matrix
│   └── sample_predictions/                # Visual predictions
└── forecasting/
    ├── metrics.json                       # Forecasting metrics
    ├── predictions_vs_actual.png          # Comparison plot
    └── risk_distribution.png              # Risk level distribution
```

### Logs
```
logs/
├── training_detection_TIMESTAMP.log       # Detection training log
├── training_forecasting_TIMESTAMP.log     # Forecasting training log
├── evaluation_detection_TIMESTAMP.log     # Detection eval log
└── evaluation_forecasting_TIMESTAMP.log   # Forecasting eval log
```

### Documentation
```
docs/
├── model_cards/
│   ├── detection_model_card.md            # Detection model info
│   └── forecasting_model_card.md          # Forecasting model info
├── training_report.pdf                    # Training summary
└── deployment_guide.md                    # Deployment instructions
```

---

## 🎯 Quick Start for Data Collection

### 1. Get Kaggle API Token
```bash
# Install Kaggle CLI
pip install kaggle

# Place API token
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset
```bash
# FloodNet (GitHub)
git clone https://github.com/BinaLab/FloodNet.git data/raw/floodnet

# OR AI4Floods (Kaggle)
kaggle datasets download -d ratthachat/ai4floods
```

### 3. Get Weather API Key
```bash
# Sign up at: https://openweathermap.org/api
# Free tier: 1000 calls/day
# Add to .env file:
echo "OPENWEATHER_API_KEY=your_key_here" > .env
```

### 4. Start Training
```bash
# Prepare data
python utils/prepare_dataset.py

# Train detection
python train_detection.py --config configs/detection_config.yaml

# Train forecasting
python train_forecasting.py --config configs/forecast_config.yaml
```

---

## 📞 Support & Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See README.md and QUICKSTART.md
- **Model Zoo**: Pretrained weights (to be added after training)
- **Tutorials**: Jupyter notebooks in `notebooks/`

---

**Last Updated**: 2025-10-11  
**Status**: Models coded, awaiting dataset collection and training  
**Next Steps**: Download datasets → Preprocess → Train → Evaluate → Deploy
