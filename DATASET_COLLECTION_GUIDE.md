# Dataset Collection Quick Reference Guide

## 🚀 Quick Start - Get Training Data in 3 Steps

### Step 1: Download Flood Detection Dataset (15-30 minutes)

**Option A: FloodNet (Recommended - Free)**
```bash
# Clone from GitHub
python utils/download_datasets.py --dataset floodnet
```

**Option B: AI4Floods from Kaggle**
```bash
# Setup Kaggle API first
pip install kaggle

# Get API key from https://www.kaggle.com/account
mkdir -p ~/.kaggle
# Download kaggle.json and move to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python utils/download_datasets.py --dataset ai4floods
```

### Step 2: Prepare Dataset (5-10 minutes)
```bash
# Resize images and organize structure
python utils/prepare_dataset.py \
  --input data/raw/floodnet \
  --output data/processed \
  --size 512 512 \
  --split

# This creates:
# data/train/images/, data/train/masks/
# data/val/images/, data/val/masks/
# data/test/images/, data/test/masks/
```

### Step 3: Download Weather Data (2 minutes)
```bash
# Generate synthetic weather data for testing
python utils/download_weather.py --synthetic \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --output data/weather/weather_data.csv

# OR use real weather data (requires API key)
python utils/download_weather.py --forecast \
  --api-key YOUR_OPENWEATHER_KEY \
  --location "Mumbai,IN" \
  --output data/weather/weather_data.csv
```

---

## 📊 Dataset Sources Summary

### Detection Dataset Sources

| Dataset | Size | Quality | License | Download |
|---------|------|---------|---------|----------|
| **FloodNet** | 2,343 images | High | Research | `python utils/download_datasets.py --dataset floodnet` |
| **AI4Floods** | 5,000+ images | High | CC BY 4.0 | Requires Kaggle API |
| **xView2** | 10,000+ images | High | CC BY-NC-SA 4.0 | Manual download from xview2.org |
| **Synthetic** | Unlimited | Medium | N/A | `python utils/create_sample_dataset.py` |

### Weather Data Sources

| Source | Coverage | Update Freq | Cost | Best For |
|--------|----------|-------------|------|----------|
| **OpenWeatherMap** | Global | Hourly | Free tier | Real-time + 5-day forecast |
| **IMD India** | India | Daily | Free | Historical Indian weather |
| **NASA POWER** | Global | Daily | Free | Historical research data |
| **Synthetic** | Custom | Any | Free | Testing/prototyping |

---

## 🔧 Detailed Instructions

### A. FloodNet Dataset Setup

**What is FloodNet?**
- 2,343 high-resolution images (3000×4000 pixels)
- Labeled for flood detection and damage assessment
- Captured from UAVs during Hurricane Harvey
- Multi-class labels (can be converted to binary)

**Download:**
```bash
git clone https://github.com/BinaLab/FloodNet.git data/raw/floodnet
```

**Structure:**
```
FloodNet/
├── train/
│   ├── train-org-img/          # Original images
│   └── train-label-img/        # Segmentation masks
├── test/
│   ├── test-org-img/
│   └── test-label-img/
└── README.md
```

**Preprocess:**
```bash
python utils/prepare_dataset.py \
  --input data/raw/floodnet/train/train-org-img \
  --output data/processed
```

### B. AI4Floods Dataset (Kaggle)

**Setup Kaggle API:**
```bash
# 1. Install Kaggle
pip install kaggle

# 2. Get API credentials
# Go to: https://www.kaggle.com/account
# Click: "Create New API Token"
# This downloads: kaggle.json

# 3. Setup credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Test connection
kaggle datasets list
```

**Download AI4Floods:**
```bash
kaggle datasets download -d ratthachat/ai4floods
unzip ai4floods.zip -d data/raw/ai4floods
```

### C. Weather Data Collection

#### Option 1: OpenWeatherMap API (Real Data)

**Get API Key:**
1. Sign up at https://openweathermap.org/price (Free tier available)
2. Go to API Keys section
3. Copy your API key

**Download Forecast:**
```bash
python utils/download_weather.py \
  --forecast \
  --api-key "YOUR_API_KEY_HERE" \
  --location "Mumbai,IN" \
  --output data/weather/forecast.csv
```

**Supported Locations:**
- Format: "City,CountryCode"
- Examples: "Mumbai,IN", "New York,US", "London,GB"

#### Option 2: Synthetic Weather Data (Testing)

**Generate Test Data:**
```bash
# 1 year of hourly data
python utils/download_weather.py \
  --synthetic \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --interval 1 \
  --output data/weather/synthetic_weather.csv
```

**Output Columns:**
- timestamp
- temperature_c
- humidity_percent
- rainfall_mm
- wind_speed_mps
- pressure_hpa
- location

---

## 📁 Expected Final Structure

After completing all steps:

```
data/
├── train/
│   ├── images/                 # 70% of dataset
│   │   ├── image_0001.jpg
│   │   ├── image_0002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_0001.png
│       ├── image_0002.png
│       └── ...
│
├── val/
│   ├── images/                 # 20% of dataset
│   └── masks/
│
├── test/
│   ├── images/                 # 10% of dataset
│   └── masks/
│
└── weather/
    ├── weather_data.csv        # Weather time series
    └── forecast.csv            # Optional forecast data
```

---

## ✅ Verification Checklist

### Detection Dataset
- [ ] Downloaded flood dataset (FloodNet or AI4Floods)
- [ ] Images resized to 512×512 pixels
- [ ] Train/val/test split created (70/20/10)
- [ ] Masks are binary (0 or 255)
- [ ] At least 1000 training images
- [ ] Image-mask pairs match by filename

**Verify:**
```bash
ls data/train/images/ | wc -l      # Should show ~70% of total
ls data/val/images/ | wc -l        # Should show ~20% of total
ls data/test/images/ | wc -l       # Should show ~10% of total
```

### Weather Dataset
- [ ] Weather data CSV created
- [ ] Contains required columns (timestamp, rainfall, humidity, temperature)
- [ ] At least 6 months of data
- [ ] Timestamps align with video/image capture times (if real data)

**Verify:**
```bash
head -5 data/weather/weather_data.csv
wc -l data/weather/weather_data.csv
```

---

## 🎯 Training Readiness Test

Run this command to verify everything is ready:

```bash
# Test data loading
python -c "
from src.data_ingestion import CCTVDataLoader
import pandas as pd

# Test detection data
try:
    loader = CCTVDataLoader('data/train')
    print(f'✅ Train images: {len(loader)} found')
except Exception as e:
    print(f'❌ Error loading train data: {e}')

# Test weather data
try:
    weather = pd.read_csv('data/weather/weather_data.csv')
    print(f'✅ Weather records: {len(weather)} found')
except Exception as e:
    print(f'❌ Error loading weather data: {e}')
"
```

**Expected Output:**
```
✅ Train images: 1500+ found
✅ Weather records: 8000+ found
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "git: command not found"
```bash
# macOS
brew install git

# Linux
sudo apt-get install git
```

### Issue 2: "kaggle: command not found"
```bash
pip install kaggle
# Make sure ~/.local/bin is in your PATH
```

### Issue 3: "Could not find a version that satisfies the requirement kaggle"
```bash
pip install --upgrade pip
pip install kaggle
```

### Issue 4: Kaggle API returns 403 Forbidden
- Make sure kaggle.json is in ~/.kaggle/
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Accept dataset terms on Kaggle website first

### Issue 5: Images and masks don't match
```bash
# Verify filenames match
ls data/train/images/ > images.txt
ls data/train/masks/ > masks.txt
diff images.txt masks.txt
```

---

## 📚 Additional Resources

### Dataset Papers & Documentation

**FloodNet:**
- Paper: https://arxiv.org/abs/2012.02951
- GitHub: https://github.com/BinaLab/FloodNet

**AI4Floods:**
- Kaggle: https://www.kaggle.com/datasets/ratthachat/ai4floods
- Related work: Google Flood Forecasting

### Weather Data APIs

**OpenWeatherMap:**
- Docs: https://openweathermap.org/api
- Pricing: https://openweathermap.org/price
- Free tier: 1,000 calls/day, 5-day forecast

**IMD India:**
- Portal: https://mausam.imd.gov.in/
- Open data: https://www.imd.gov.in/pages/services_opendata.php

**NASA POWER:**
- Portal: https://power.larc.nasa.gov/
- API Docs: https://power.larc.nasa.gov/docs/

---

## 🎓 Next Steps After Data Collection

Once you have the datasets ready:

### 1. Train Detection Model
```bash
python train_detection.py \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --epochs 50
```

### 2. Train Forecasting Model
```bash
python train_forecasting.py \
  --train data/weather/train.csv \
  --val data/weather/val.csv \
  --epochs 100
```

### 3. Run Full Pipeline Demo
```bash
python demo.py \
  --video test_video.mp4 \
  --detection-model models/detection/best_model.pth \
  --forecast-model models/forecasting/best_model.pth
```

---

## 📞 Need Help?

- **GitHub Issues**: Open an issue in the repository
- **Documentation**: See README.md and MODEL_TRAINING_REQUIREMENTS.md
- **Quick Start**: See QUICKSTART.md

---

**Last Updated:** 2025-10-11  
**Status:** Ready for data collection
