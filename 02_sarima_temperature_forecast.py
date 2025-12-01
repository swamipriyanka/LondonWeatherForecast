"""
SARIMA Model for London Temperature Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*70)
print("SARIMA Temperature Forecasting Model")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

df = pd.read_csv('data/processed/london_weather_with_features.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Resample to DAILY average
daily_temp = df['temperature'].resample('D').mean()

print(f"✅ Data loaded and resampled to daily!")
print(f"Total days: {len(daily_temp):,}")
print(f"Date range: {daily_temp.index.min()} to {daily_temp.index.max()}")

# ============================================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

# Use 80% for training, 20% for testing
train_size = int(len(daily_temp) * 0.8)
train_data = daily_temp[:train_size]
test_data = daily_temp[train_size:]

print(f"\nTraining set: {len(train_data)} days")
print(f"  From: {train_data.index.min().date()}")
print(f"  To: {train_data.index.max().date()}")
print(f"\nTest set: {len(test_data)} days")
print(f"  From: {test_data.index.min().date()}")
print(f"  To: {test_data.index.max().date()}")

# Visualize the split
plt.figure(figsize=(15, 5))
plt.plot(train_data.index, train_data.values, label='Training Data', color='blue', linewidth=1)
plt.plot(test_data.index, test_data.values, label='Test Data', color='orange', linewidth=1)
plt.axvline(x=train_data.index[-1], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Train-Test Split for Daily Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/04_train_test_split.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: visualizations/04_train_test_split.png")
plt.show()

# ============================================================================
# STEP 3: SEASONAL DECOMPOSITION
# ============================================================================

print("\n" + "="*70)
print("SEASONAL DECOMPOSITION")
print("="*70)

print("\nDecomposing time series into components...")
# Use yearly seasonality (365 days)
decomposition = seasonal_decompose(train_data, model='additive', period=30)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 10))
fig.suptitle('Seasonal Decomposition of Daily Temperature', fontsize=16, fontweight='bold')

decomposition.observed.plot(ax=axes[0], color='blue', linewidth=1)
axes[0].set_ylabel('Observed')
axes[0].set_title('Original Time Series')

decomposition.trend.plot(ax=axes[1], color='green', linewidth=1.5)
axes[1].set_ylabel('Trend')
axes[1].set_title('Trend Component')

decomposition.seasonal.plot(ax=axes[2], color='orange', linewidth=1)
axes[2].set_ylabel('Seasonal')
axes[2].set_title('Seasonal Component (Yearly Pattern)')

decomposition.resid.plot(ax=axes[3], color='red', linewidth=1)
axes[3].set_ylabel('Residual')
axes[3].set_title('Residual (Random) Component')

plt.tight_layout()
plt.savefig('visualizations/05_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/05_seasonal_decomposition.png")
plt.show()

# ============================================================================
# STEP 4: BUILD SARIMA MODEL
# ============================================================================

print("\n" + "="*70)
print("BUILDING SARIMA MODEL")
print("="*70)

print("\nFitting SARIMA model...")
print("Parameters: SARIMA(1,0,1)x(1,1,0,7)")


# Simpler SARIMA parameters for daily data
# (p,d,q) - non-seasonal: (1,0,1)
# (P,D,Q,s) - seasonal: (1,1,0,7) - weekly pattern

model = SARIMAX(
    train_data,
    order=(1, 0, 1),           # (p,d,q) - simpler non-seasonal
    seasonal_order=(1, 1, 0, 7),  # (P,D,Q,s) - weekly seasonality
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit the model
sarima_model = model.fit(disp=False, maxiter=100)

print("\n✅ Model fitted successfully!")

# ============================================================================
# STEP 5: MAKE PREDICTIONS ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

print("\nGenerating forecasts for test period...")

# Get predictions - this time doing it step-by-step to avoid divergence
predictions = []
history = train_data.copy()

for i in range(len(test_data)):
    # Fit model on all data so far
    if i % 50 == 0:  # Print progress every 50 days
        print(f"  Predicting day {i+1}/{len(test_data)}...")
    
    model_temp = SARIMAX(
        history,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model_temp.fit(disp=False, maxiter=50, method='nm')
    
    # Forecast next day
    forecast = model_fit.forecast(steps=1)
    predictions.append(forecast.iloc[0])
    
    # Add actual value to history for next iteration
    history = pd.concat([history, test_data.iloc[i:i+1]])

predictions = pd.Series(predictions, index=test_data.index)

print("✅ Predictions generated!")

# ============================================================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ============================================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_data, predictions))
mae = mean_absolute_error(test_data, predictions)
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

print(f"\nPerformance Metrics:")
print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}°C")
print(f"  MAE (Mean Absolute Error): {mae:.2f}°C")
print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

print(f"\nInterpretation:")
print(f"  On average, predictions are off by ±{mae:.2f}°C")
if mae < 2.5:
    print(f"  ✅ That's excellent accuracy for weather forecasting!")
elif mae < 4.0:
    print(f"  ✅ That's good accuracy for weather forecasting!")
else:
    print(f"  ✅ That's reasonable for simple SARIMA model!")

# ============================================================================
# STEP 7: VISUALIZE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Full comparison
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('SARIMA Model: Predictions vs Actual Temperature', fontsize=16, fontweight='bold')

# Full dataset
axes[0].plot(train_data.index, train_data.values, label='Training Data', 
             color='blue', linewidth=1, alpha=0.7)
axes[0].plot(test_data.index, test_data.values, label='Actual Test Data', 
             color='green', linewidth=1.5)
axes[0].plot(predictions.index, predictions.values, label='SARIMA Predictions', 
             color='red', linewidth=1.5, linestyle='--')
axes[0].axvline(x=train_data.index[-1], color='black', linestyle=':', 
                linewidth=2, label='Train/Test Split', alpha=0.5)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Full Dataset View')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoomed in on predictions (first 30 days)
zoom_days = min(30, len(test_data))
axes[1].plot(test_data.index[:zoom_days], test_data.values[:zoom_days], 
             label='Actual', color='green', linewidth=2, marker='o', markersize=4)
axes[1].plot(predictions.index[:zoom_days], predictions.values[:zoom_days], 
             label='SARIMA Predictions', color='red', linewidth=2, marker='x', markersize=4)
axes[1].fill_between(predictions.index[:zoom_days], 
                      predictions.values[:zoom_days] - rmse,
                      predictions.values[:zoom_days] + rmse,
                      alpha=0.2, color='red', label=f'±{rmse:.2f}°C Error Band')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_title('Detailed View: First 30 Days of Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_sarima_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/06_sarima_predictions.png")
plt.show()

# Plot 2: Scatter plot (Actual vs Predicted)
plt.figure(figsize=(8, 8))
plt.scatter(test_data, predictions, alpha=0.5, s=20)
plt.plot([test_data.min(), test_data.max()], 
         [test_data.min(), test_data.max()], 
         'r--', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('SARIMA: Actual vs Predicted Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/07_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/07_actual_vs_predicted.png")
plt.show()

# Plot 3: Prediction errors
errors = test_data - predictions
plt.figure(figsize=(15, 5))
plt.plot(test_data.index, errors, linewidth=1, color='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.fill_between(test_data.index, -rmse, rmse, alpha=0.2, color='red')
plt.xlabel('Date')
plt.ylabel('Prediction Error (°C)')
plt.title('SARIMA Prediction Errors Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/08_prediction_errors.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/08_prediction_errors.png")
plt.show()

# Plot 4: Error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Error: {errors.mean():.2f}°C')
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('visualizations/09_error_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/09_error_distribution.png")
plt.show()

# ============================================================================
# STEP 8: FORECAST FUTURE (NEXT 7 DAYS)
# ============================================================================

print("\n" + "="*70)
print("FORECASTING NEXT 7 DAYS")
print("="*70)

print("\nCreating forecast model on full dataset.")
full_model = SARIMAX(
    daily_temp,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
full_sarima = full_model.fit(disp=False, maxiter=100, method='nm')

# Forecast next 7 days
future_forecast = full_sarima.forecast(steps=7)

print(f"✅ Future forecast generated!")

# Plot future forecast
plt.figure(figsize=(15, 6))
plt.plot(daily_temp.index[-30:], daily_temp.values[-30:], 
         label='Historical (Last 30 Days)', color='blue', linewidth=2)
plt.plot(future_forecast.index, future_forecast.values, 
         label='Forecast (Next 7 Days)', color='red', linewidth=2.5, 
         marker='o', markersize=8, linestyle='--')
plt.axvline(x=daily_temp.index[-1], color='green', linestyle=':', linewidth=2, 
            label='Forecast Start', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('7-Day Temperature Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/10_future_forecast.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/10_future_forecast.png")
plt.show()

# Print forecast summary
print("\n7-Day Forecast:")
for date, temp in future_forecast.items():
    print(f"  {date.strftime('%A, %Y-%m-%d')}: {temp:.1f}°C")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save predictions
results_df = pd.DataFrame({
    'date': test_data.index,
    'actual': test_data.values,
    'predicted': predictions.values,
    'error': errors.values
})
results_df.to_csv('data/processed/sarima_predictions.csv', index=False)
print("✅ Saved predictions: data/processed/sarima_predictions.csv")

# Save future forecast
future_df = pd.DataFrame({
    'date': future_forecast.index,
    'forecast': future_forecast.values
})
future_df.to_csv('data/processed/future_forecast_7days.csv', index=False)
print("✅ Saved future forecast: data/processed/future_forecast_7days.csv")

print("\n" + "="*70)
print("SARIMA MODEL COMPLETE!")
print("="*70)

print(f"\nSummary:")
print(f"  Model: SARIMA(1,0,1)x(1,1,0,7) on daily data")
print(f"  Accuracy: MAE = {mae:.2f}°C, RMSE = {rmse:.2f}°C")
print(f"  Visualizations: 7 charts created")
print(f"  Future forecast: Next 7 days")

