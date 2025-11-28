import requests

def get_training_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,rain,weather_code,surface_pressure,cloud_cover,wind_speed_10m",
        "past_days": 93  # Get last 3 months of data for training
    }
    response = requests.get(url, params=params)
    return response.json()

if __name__ == "__main__":
    latitude = 11.3655
    longitude = 76.7843
    
    data = get_training_data(latitude, longitude)
    print(data)