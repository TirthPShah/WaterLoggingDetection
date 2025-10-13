"""
Utility script to download historical weather data from OpenWeatherMap API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time
import json
from typing import Optional


class WeatherDataDownloader:
    """Download weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        """
        Initialize weather data downloader
        
        Args:
            api_key: OpenWeatherMap API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.history_url = "https://history.openweathermap.org/data/2.5/history/city"
    
    def get_current_weather(self, location: str) -> dict:
        """
        Get current weather data
        
        Args:
            location: City name (e.g., "Mumbai,IN")
        
        Returns:
            Weather data dictionary
        """
        url = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def get_forecast(self, location: str, days: int = 5) -> dict:
        """
        Get weather forecast
        
        Args:
            location: City name
            days: Number of days (max 5 for free tier)
        
        Returns:
            Forecast data dictionary
        """
        url = f"{self.base_url}/forecast"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def parse_weather_data(self, data: dict) -> dict:
        """
        Parse weather data into standard format
        
        Args:
            data: Raw API response
        
        Returns:
            Parsed weather data
        """
        weather_info = {
            'timestamp': datetime.fromtimestamp(data['dt']),
            'temperature_c': data['main']['temp'],
            'humidity_percent': data['main']['humidity'],
            'pressure_hpa': data['main']['pressure'],
            'rainfall_mm': data.get('rain', {}).get('1h', 0.0),
            'wind_speed_mps': data['wind']['speed'],
            'clouds_percent': data['clouds']['all'],
            'weather_description': data['weather'][0]['description']
        }
        
        return weather_info
    
    def download_forecast_data(
        self, 
        location: str,
        output_file: str,
        days: int = 5
    ):
        """
        Download and save forecast data
        
        Args:
            location: City name
            output_file: Output CSV file path
            days: Number of forecast days
        """
        print(f"Downloading {days}-day forecast for {location}...")
        
        forecast_data = self.get_forecast(location, days)
        
        # Parse forecast entries
        weather_records = []
        for entry in forecast_data['list']:
            weather_info = self.parse_weather_data(entry)
            weather_records.append(weather_info)
        
        # Create DataFrame
        df = pd.DataFrame(weather_records)
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(weather_records)} forecast records to {output_file}")
        return df


def generate_synthetic_weather_data(
    start_date: str,
    end_date: str,
    output_file: str,
    interval_hours: int = 1,
    location: str = "Mumbai"
):
    """
    Generate synthetic weather data for testing (when API is not available)
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Output CSV file
        interval_hours: Time interval in hours
        location: Location name (for metadata)
    """
    import numpy as np
    
    print(f"Generating synthetic weather data from {start_date} to {end_date}...")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate timestamps
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(hours=interval_hours)
    
    num_records = len(timestamps)
    
    # Generate synthetic data with realistic patterns
    # Temperature: 20-35°C with daily cycle
    base_temp = 27
    temp_variation = 7
    temperatures = base_temp + temp_variation * np.sin(
        np.arange(num_records) * 2 * np.pi / (24 / interval_hours)
    ) + np.random.normal(0, 2, num_records)
    
    # Humidity: 60-90% inversely correlated with temperature
    humidities = 75 + 15 * np.sin(
        np.arange(num_records) * 2 * np.pi / (24 / interval_hours) + np.pi
    ) + np.random.normal(0, 5, num_records)
    humidities = np.clip(humidities, 40, 95)
    
    # Rainfall: mostly 0, occasional rain events
    rainfall = np.zeros(num_records)
    rain_events = np.random.choice(num_records, size=int(num_records * 0.15), replace=False)
    for idx in rain_events:
        # Rain duration 3-12 hours
        duration = np.random.randint(3, 12)
        intensity = np.random.exponential(5)  # mm/hour
        for i in range(duration):
            if idx + i < num_records:
                rainfall[idx + i] = max(0, intensity + np.random.normal(0, 2))
    
    # Wind speed: 2-15 m/s
    wind_speeds = 5 + 3 * np.random.randn(num_records)
    wind_speeds = np.clip(wind_speeds, 0, 20)
    
    # Pressure: 1000-1020 hPa
    pressures = 1010 + 5 * np.sin(
        np.arange(num_records) * 2 * np.pi / (48 / interval_hours)
    ) + np.random.normal(0, 2, num_records)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_c': temperatures,
        'humidity_percent': humidities,
        'rainfall_mm': rainfall,
        'wind_speed_mps': wind_speeds,
        'pressure_hpa': pressures,
        'location': location
    })
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated {len(df)} synthetic weather records")
    print(f"   Saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"   Temperature: {df['temperature_c'].min():.1f}°C - {df['temperature_c'].max():.1f}°C")
    print(f"   Humidity: {df['humidity_percent'].min():.1f}% - {df['humidity_percent'].max():.1f}%")
    print(f"   Total rainfall: {df['rainfall_mm'].sum():.1f} mm")
    print(f"   Rain events: {(df['rainfall_mm'] > 0).sum()} hours")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download historical weather data"
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenWeatherMap API key'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='Mumbai,IN',
        help='Location (e.g., "Mumbai,IN")'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD) for synthetic data'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD) for synthetic data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/weather/weather_data.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--forecast',
        action='store_true',
        help='Download forecast data (requires API key)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic weather data'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Time interval in hours (for synthetic data)'
    )
    
    args = parser.parse_args()
    
    if args.synthetic:
        # Generate synthetic data
        if not args.start_date or not args.end_date:
            # Default to 6 months of data
            end = datetime.now()
            start = end - timedelta(days=180)
            args.start_date = start.strftime('%Y-%m-%d')
            args.end_date = end.strftime('%Y-%m-%d')
        
        generate_synthetic_weather_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.output,
            interval_hours=args.interval,
            location=args.location
        )
    
    elif args.forecast:
        # Download forecast data
        if not args.api_key:
            print("Error: --api-key required for forecast download")
            print("Get a free API key at: https://openweathermap.org/api")
            return
        
        downloader = WeatherDataDownloader(args.api_key)
        downloader.download_forecast_data(
            location=args.location,
            output_file=args.output
        )
    
    else:
        print("Please specify --synthetic or --forecast mode")
        print("\nExamples:")
        print("  # Generate synthetic data")
        print("  python download_weather.py --synthetic --start-date 2023-01-01 --end-date 2024-01-01")
        print("\n  # Download forecast (requires API key)")
        print("  python download_weather.py --forecast --api-key YOUR_KEY --location 'Mumbai,IN'")


if __name__ == "__main__":
    main()
