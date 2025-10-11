"""
Data ingestion module for CCTV images, video frames, and weather data
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator, Tuple, Optional, List, Dict
import json
from datetime import datetime, timedelta


class CCTVDataLoader:
    """Load CCTV images or video frames"""
    
    def __init__(self, source_path: str, frame_skip: int = 1):
        """
        Initialize CCTV data loader
        
        Args:
            source_path: Path to image directory or video file
            frame_skip: Process every Nth frame (for videos)
        """
        self.source_path = Path(source_path)
        self.frame_skip = frame_skip
        self.is_video = self._check_if_video()
        
    def _check_if_video(self) -> bool:
        """Check if source is a video file or image directory"""
        if self.source_path.is_file():
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            return self.source_path.suffix.lower() in video_extensions
        return False
    
    def load_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        Load frames from video or image directory
        
        Yields:
            Tuple of (frame, metadata)
            - frame: numpy array (H, W, C) in BGR format
            - metadata: dict with timestamp, frame_number, source info
        """
        if self.is_video:
            yield from self._load_video_frames()
        else:
            yield from self._load_image_frames()
    
    def _load_video_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(str(self.source_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.source_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                timestamp = datetime.now() - timedelta(
                    seconds=(total_frames - frame_count) / fps
                )
                
                metadata = {
                    'frame_number': frame_count,
                    'timestamp': timestamp.isoformat(),
                    'source': str(self.source_path),
                    'fps': fps,
                    'processed_frame_index': processed_count
                }
                
                yield frame, metadata
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
    
    def _load_image_frames(self) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """Load frames from image directory"""
        if not self.source_path.is_dir():
            raise ValueError(f"Source path is not a directory: {self.source_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = sorted([
            f for f in self.source_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No images found in directory: {self.source_path}")
        
        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            
            if frame is None:
                print(f"Warning: Could not read image {img_path}, skipping...")
                continue
            
            metadata = {
                'frame_number': idx,
                'timestamp': datetime.now().isoformat(),
                'source': str(img_path),
                'processed_frame_index': idx
            }
            
            yield frame, metadata


class WeatherDataLoader:
    """Load and manage weather data"""
    
    def __init__(self, weather_data_path: Optional[str] = None):
        """
        Initialize weather data loader
        
        Args:
            weather_data_path: Path to weather data CSV/JSON file
        """
        self.weather_data_path = weather_data_path
        self.weather_df = None
        
        if weather_data_path:
            self.load_weather_data(weather_data_path)
    
    def load_weather_data(self, data_path: str):
        """
        Load weather data from file
        
        Expected format (CSV):
        - timestamp, rainfall_mm, humidity_percent, temperature_c, ...
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            self.weather_df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            self.weather_df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Convert timestamp to datetime
        if 'timestamp' in self.weather_df.columns:
            self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
            self.weather_df.sort_values('timestamp', inplace=True)
    
    def get_weather_at_timestamp(
        self, 
        timestamp: str, 
        history_hours: int = 6
    ) -> Dict[str, float]:
        """
        Get weather data at specific timestamp with historical context
        
        Args:
            timestamp: ISO format timestamp
            history_hours: Hours of historical data to include
            
        Returns:
            Dictionary with current and historical weather features
        """
        if self.weather_df is None:
            # Return dummy data if no weather data available
            return self._get_dummy_weather_data()
        
        target_time = pd.to_datetime(timestamp)
        start_time = target_time - pd.Timedelta(hours=history_hours)
        
        # Filter data within time range
        mask = (self.weather_df['timestamp'] >= start_time) & \
               (self.weather_df['timestamp'] <= target_time)
        relevant_data = self.weather_df[mask]
        
        if relevant_data.empty:
            return self._get_dummy_weather_data()
        
        # Extract features
        weather_features = {
            'current_rainfall_mm': relevant_data['rainfall_mm'].iloc[-1] if 'rainfall_mm' in relevant_data.columns else 0.0,
            'current_humidity_percent': relevant_data['humidity_percent'].iloc[-1] if 'humidity_percent' in relevant_data.columns else 50.0,
            'current_temperature_c': relevant_data['temperature_c'].iloc[-1] if 'temperature_c' in relevant_data.columns else 25.0,
            'avg_rainfall_6h': relevant_data['rainfall_mm'].mean() if 'rainfall_mm' in relevant_data.columns else 0.0,
            'max_rainfall_6h': relevant_data['rainfall_mm'].max() if 'rainfall_mm' in relevant_data.columns else 0.0,
            'cumulative_rainfall_6h': relevant_data['rainfall_mm'].sum() if 'rainfall_mm' in relevant_data.columns else 0.0,
            'avg_humidity_6h': relevant_data['humidity_percent'].mean() if 'humidity_percent' in relevant_data.columns else 50.0,
        }
        
        return weather_features
    
    def _get_dummy_weather_data(self) -> Dict[str, float]:
        """Return dummy weather data when actual data is unavailable"""
        return {
            'current_rainfall_mm': 0.0,
            'current_humidity_percent': 50.0,
            'current_temperature_c': 25.0,
            'avg_rainfall_6h': 0.0,
            'max_rainfall_6h': 0.0,
            'cumulative_rainfall_6h': 0.0,
            'avg_humidity_6h': 50.0,
        }
    
    def create_sample_weather_data(self, output_path: str, num_records: int = 100):
        """
        Create sample weather data for testing
        
        Args:
            output_path: Path to save sample data
            num_records: Number of records to generate
        """
        timestamps = [
            datetime.now() - timedelta(minutes=i*10) 
            for i in range(num_records-1, -1, -1)
        ]
        
        # Generate synthetic weather data
        np.random.seed(42)
        data = {
            'timestamp': timestamps,
            'rainfall_mm': np.random.exponential(2, num_records),
            'humidity_percent': np.random.uniform(40, 95, num_records),
            'temperature_c': np.random.normal(25, 5, num_records),
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Sample weather data saved to {output_path}")
