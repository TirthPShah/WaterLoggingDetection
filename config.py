"""
Configuration file for AI-CCTV Waterlogging Detection & Forecasting System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Image processing settings
IMAGE_SIZE = (512, 512)  # (height, width)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Detection model settings
DETECTION_MODEL_NAME = "unet"  # Options: unet, deeplabv3, fpn
ENCODER_NAME = "resnet34"  # Encoder backbone
ENCODER_WEIGHTS = "imagenet"
DETECTION_THRESHOLD = 0.5  # Probability threshold for waterlogging
DEVICE = "cuda"  # or "cpu"

# Postprocessing settings
MORPH_KERNEL_SIZE = 5
MIN_WATERLOG_AREA = 100  # Minimum area in pixels to consider as waterlogging

# Weather data settings
WEATHER_FEATURES = ["rainfall_mm", "humidity_percent", "temperature_c"]
WEATHER_HISTORY_HOURS = 6  # Hours of historical weather data to use

# Forecasting settings
FORECAST_HORIZON_MINUTES = 30  # Predict waterlogging risk for next N minutes
FORECAST_MODEL_TYPE = "lstm"  # Options: lstm, random_forest, gradient_boosting
SEQUENCE_LENGTH = 10  # Number of timesteps to use for prediction

# Fusion settings
DETECTION_WEIGHT = 0.6  # Weight for detection model
FORECAST_WEIGHT = 0.4   # Weight for forecasting model
TEMPORAL_SMOOTHING_ALPHA = 0.3  # Exponential smoothing factor

# Visualization settings
OVERLAY_ALPHA = 0.4  # Transparency of overlay mask
COLORMAP = "jet"  # Colormap for risk visualization
RISK_LEVELS = {
    "low": (0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 1.0)
}

# Video processing settings
VIDEO_FPS = 30
FRAME_SKIP = 5  # Process every Nth frame

# Export settings
EXPORT_FORMAT = "json"  # Options: json, csv
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
