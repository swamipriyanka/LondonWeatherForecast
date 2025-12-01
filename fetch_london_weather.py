"""
Script to fetch historical weather data for London
Uses Open-Meteo API (free, no API key required)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_london_weather(start_date, end_date, save_path='data/raw/london_weather.csv'):
    """
    Fetch historical weather data for London from Open-Meteo API
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    save_path : str
        Path where to save the CSV file
    """
    
    print("Fetching London weather data...")
    print(f"Period: {start_date} to {end_date}")
    
    # London coordinates
    latitude = 51.5074
    longitude = -0.1278
    
    # API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover"
        ],
        "timezone": "Europe/London"
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error for bad status codes
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'precipitation': data['hourly']['precipitation'],
            'pressure': data['hourly']['pressure_msl'],
            'wind_speed': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'cloud_cover': data['hourly']['cloud_cover']
        })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        print(f"\n✅ Success! Data saved to {save_path}")
        print(f"\nDataset Info:")
        print(f"- Total records: {len(df):,}")
        print(f"- Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"- Features: {', '.join(df.columns[1:])}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(df.describe())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing values detected:")
            print(missing[missing > 0])
        else:
            print(f"\n✅ No missing values!")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


if __name__ == "__main__":
    # Fetch 2 years of data
    # You can adjust these dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years ago
    
    print("="*60)
    print("London Weather Data Collection")
    print("="*60)
    
    df = fetch_london_weather(start_date, end_date)
    
    if df is not None:
        print("\n" + "="*60)
        print("Data collection complete! Next steps:")
        print("="*60)
        print("1. Check the data: data/raw/london_weather.csv")
        print("2. Run exploratory data analysis")
        print("3. Start building your first model!")
        print("\nHappy forecasting! 🌦️")