"""
Exploratory Data Analysis for London Weather Data
This script analyzes the weather data and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("London Weather Data - Exploratory Data Analysis")
print("="*70)

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================


df = pd.read_csv('data/raw/london_weather.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"✅ Data loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ============================================================================
# STEP 2: BASIC DATA INFO
# ============================================================================

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)

print("\nColumn Information:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✅ No missing values!")

# ============================================================================
# STEP 3: TIME SERIES ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("TIME SERIES VISUALIZATIONS")
print("="*70)

# Create a figure with multiple subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle('London Weather Time Series (Full Dataset)', fontsize=16, fontweight='bold')

# Plot 1: Temperature
axes[0].plot(df['datetime'], df['temperature'], color='orangered', linewidth=0.5)
axes[0].set_ylabel('Temperature (°C)', fontsize=12)
axes[0].set_title('Temperature Over Time')
axes[0].grid(True, alpha=0.3)

# Plot 2: Humidity
axes[1].plot(df['datetime'], df['humidity'], color='steelblue', linewidth=0.5)
axes[1].set_ylabel('Humidity (%)', fontsize=12)
axes[1].set_title('Humidity Over Time')
axes[1].grid(True, alpha=0.3)

# Plot 3: Precipitation
axes[2].plot(df['datetime'], df['precipitation'], color='darkblue', linewidth=0.5)
axes[2].set_ylabel('Precipitation (mm)', fontsize=12)
axes[2].set_title('Precipitation Over Time')
axes[2].grid(True, alpha=0.3)

# Plot 4: Wind Speed
axes[3].plot(df['datetime'], df['wind_speed'], color='green', linewidth=0.5)
axes[3].set_ylabel('Wind Speed (km/h)', fontsize=12)
axes[3].set_xlabel('Date', fontsize=12)
axes[3].set_title('Wind Speed Over Time')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/01_time_series_full.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/01_time_series_full.png")
plt.show()

# ============================================================================
# STEP 4: SEASONAL PATTERNS
# ============================================================================

print("\n" + "="*70)
print("SEASONAL PATTERNS")
print("="*70)

# Extract date components
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month_name'] = df['datetime'].dt.month_name()

# Monthly average temperature
monthly_temp = df.groupby('month')['temperature'].agg(['mean', 'min', 'max'])
print("\nMonthly Temperature Statistics:")
print(monthly_temp.round(2))

# Create seasonal visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('London Weather Seasonal Patterns', fontsize=16, fontweight='bold')

# Plot 1: Monthly Average Temperature
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 0].plot(monthly_temp.index, monthly_temp['mean'], marker='o', 
                linewidth=2, markersize=8, color='orangered')
axes[0, 0].fill_between(monthly_temp.index, monthly_temp['min'], 
                         monthly_temp['max'], alpha=0.3, color='orange')
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(month_names)
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].set_title('Monthly Temperature (Mean with Min/Max Range)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Hourly Average Temperature
hourly_temp = df.groupby('hour')['temperature'].mean()
axes[0, 1].plot(hourly_temp.index, hourly_temp.values, marker='o', 
                linewidth=2, markersize=6, color='coral')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].set_title('Average Temperature by Hour of Day')
axes[0, 1].set_xticks(range(0, 24, 2))
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Temperature Distribution
axes[1, 0].hist(df['temperature'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(df['temperature'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {df['temperature'].mean():.1f}°C")
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Temperature Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Monthly Precipitation
monthly_precip = df.groupby('month')['precipitation'].sum()
axes[1, 1].bar(monthly_precip.index, monthly_precip.values, color='steelblue', alpha=0.7)
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].set_xticklabels(month_names)
axes[1, 1].set_ylabel('Total Precipitation (mm)')
axes[1, 1].set_title('Total Precipitation by Month')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/02_seasonal_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/02_seasonal_patterns.png")
plt.show()

# ============================================================================
# STEP 5: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Select numeric columns for correlation
numeric_cols = ['temperature', 'humidity', 'precipitation', 
                'pressure', 'wind_speed', 'cloud_cover']

correlation_matrix = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Between Weather Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/03_correlation_heatmap.png")
plt.show()

# ============================================================================
# STEP 6: KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Temperature:")
print(f"   - Average: {df['temperature'].mean():.2f}°C")
print(f"   - Minimum: {df['temperature'].min():.2f}°C")
print(f"   - Maximum: {df['temperature'].max():.2f}°C")
print(f"   - Standard Deviation: {df['temperature'].std():.2f}°C")

print(f"\n2. Humidity:")
print(f"   - Average: {df['humidity'].mean():.2f}%")

print(f"\n3. Precipitation:")
print(f"   - Total: {df['precipitation'].sum():.2f} mm")
print(f"   - Days with rain: {(df['precipitation'] > 0).sum()} hours")

print(f"\n4. Wind:")
print(f"   - Average speed: {df['wind_speed'].mean():.2f} km/h")
print(f"   - Maximum speed: {df['wind_speed'].max():.2f} km/h")

# Find coldest and hottest days
coldest_day = df.loc[df['temperature'].idxmin()]
hottest_day = df.loc[df['temperature'].idxmax()]

print(f"\n5. Extremes:")
print(f"   - Coldest moment: {coldest_day['datetime']} ({coldest_day['temperature']:.2f}°C)")
print(f"   - Hottest moment: {hottest_day['datetime']} ({hottest_day['temperature']:.2f}°C)")

# ============================================================================
# STEP 7: SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

summary = pd.DataFrame({
    'Feature': numeric_cols,
    'Mean': [df[col].mean() for col in numeric_cols],
    'Std': [df[col].std() for col in numeric_cols],
    'Min': [df[col].min() for col in numeric_cols],
    'Max': [df[col].max() for col in numeric_cols]
})

print("\n", summary.round(2))

# ============================================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*70)
print("SAVING PROCESSED DATA")
print("="*70)

# Save the data with additional features
df.to_csv('data/processed/london_weather_with_features.csv', index=False)
print("✅ Saved: data/processed/london_weather_with_features.csv")

