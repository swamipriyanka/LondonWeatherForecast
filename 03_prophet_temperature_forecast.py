"""
Prophet Model for London Temperature Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*70)
print("Prophet Temperature Forecasting Model")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

df = pd.read_csv('data/processed/london_weather_with_features.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Prophet requires specific column names: 'ds' (date) and 'y' (value)
prophet_df = df[['datetime', 'temperature']].copy()
prophet_df.columns = ['ds', 'y']

print(f"✅ Data loaded!")
print(f"Total observations: {len(prophet_df):,} hours")
print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

# ============================================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

# Use 80% for training, 20% for testing
train_size = int(len(prophet_df) * 0.8)
train_data = prophet_df[:train_size].copy()
test_data = prophet_df[train_size:].copy()

print(f"\nTraining set: {len(train_data):,} hours")
print(f"  From: {train_data['ds'].min()}")
print(f"  To: {train_data['ds'].max()}")
print(f"\nTest set: {len(test_data):,} hours")
print(f"  From: {test_data['ds'].min()}")
print(f"  To: {test_data['ds'].max()}")

# Visualize the split
plt.figure(figsize=(15, 5))
plt.plot(train_data['ds'], train_data['y'], label='Training Data', 
         color='blue', linewidth=0.5, alpha=0.7)
plt.plot(test_data['ds'], test_data['y'], label='Test Data', 
         color='orange', linewidth=0.5)
plt.axvline(x=train_data['ds'].iloc[-1], color='red', linestyle='--', 
            linewidth=2, label='Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Train-Test Split for Hourly Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/11_prophet_train_test_split.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: visualizations/11_prophet_train_test_split.png")
plt.show()

# ============================================================================
# STEP 3: BUILD PROPHET MODEL
# ============================================================================

print("\n" + "="*70)
print("BUILDING PROPHET MODEL")
print("="*70)

print("\nInitializing Prophet model...")
print("Configuring seasonality components...")

# Create Prophet model with custom settings
model = Prophet(
    yearly_seasonality=True,      # Capture yearly patterns (summer/winter)
    weekly_seasonality=True,      # Capture weekly patterns
    daily_seasonality=True,       # Capture daily patterns (day/night)
    seasonality_mode='additive',  # Additive seasonality
    changepoint_prior_scale=0.05, # Flexibility of trend changes (lower = less flexible)
    interval_width=0.95           # 95% confidence interval
)

print("✅ Model configured!")
print("\nFitting model to training data...")
print("This may take 1-2 minutes...")

# Fit the model
model.fit(train_data)

print("✅ Model fitted successfully!")

# ============================================================================
# STEP 4: MAKE PREDICTIONS ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

print("\nGenerating forecasts for test period...")

# Create dataframe with test dates
future = test_data[['ds']].copy()

# Make predictions
forecast = model.predict(future)

# Extract predictions
predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
predictions.columns = ['ds', 'prediction', 'lower_bound', 'upper_bound']

print("✅ Predictions generated!")

# ============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# ============================================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

# Merge actual and predicted values
results = test_data.merge(predictions, on='ds')

# Calculate metrics
rmse = np.sqrt(mean_squared_error(results['y'], results['prediction']))
mae = mean_absolute_error(results['y'], results['prediction'])
mape = np.mean(np.abs((results['y'] - results['prediction']) / results['y'])) * 100

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
    print(f"  ✅ That's reasonable for this model!")

# ============================================================================
# STEP 6: VISUALIZE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Full predictions with confidence intervals
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('Prophet Model: Predictions vs Actual Temperature', fontsize=16, fontweight='bold')

# Full dataset view
axes[0].plot(train_data['ds'], train_data['y'], label='Training Data', 
             color='blue', linewidth=0.5, alpha=0.7)
axes[0].plot(results['ds'], results['y'], label='Actual Test Data', 
             color='green', linewidth=1)
axes[0].plot(results['ds'], results['prediction'], label='Prophet Predictions', 
             color='red', linewidth=1, linestyle='--')
axes[0].fill_between(results['ds'], results['lower_bound'], results['upper_bound'],
                      alpha=0.2, color='red', label='95% Confidence Interval')
axes[0].axvline(x=train_data['ds'].iloc[-1], color='black', linestyle=':', 
                linewidth=2, label='Train/Test Split', alpha=0.5)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Full Dataset View')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Zoomed view (first 7 days)
zoom_hours = min(24*7, len(results))
axes[1].plot(results['ds'][:zoom_hours], results['y'][:zoom_hours], 
             label='Actual', color='green', linewidth=2, marker='o', markersize=2)
axes[1].plot(results['ds'][:zoom_hours], results['prediction'][:zoom_hours], 
             label='Prophet Predictions', color='red', linewidth=2, marker='x', markersize=2)
axes[1].fill_between(results['ds'][:zoom_hours], 
                      results['lower_bound'][:zoom_hours],
                      results['upper_bound'][:zoom_hours],
                      alpha=0.2, color='red', label='95% Confidence')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_title('Detailed View: First Week of Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/12_prophet_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/12_prophet_predictions.png")
plt.show()

# Plot 2: Scatter plot (Actual vs Predicted)
plt.figure(figsize=(8, 8))
plt.scatter(results['y'], results['prediction'], alpha=0.3, s=5)
plt.plot([results['y'].min(), results['y'].max()], 
         [results['y'].min(), results['y'].max()], 
         'r--', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Prophet: Actual vs Predicted Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/13_prophet_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/13_prophet_actual_vs_predicted.png")
plt.show()

# Plot 3: Prediction errors
errors = results['y'] - results['prediction']
plt.figure(figsize=(15, 5))
plt.plot(results['ds'], errors, linewidth=0.5, color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.fill_between(results['ds'], -rmse, rmse, alpha=0.2, color='red')
plt.xlabel('Date')
plt.ylabel('Prediction Error (°C)')
plt.title('Prophet Prediction Errors Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/14_prophet_prediction_errors.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/14_prophet_prediction_errors.png")
plt.show()

# Plot 4: Error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Error: {errors.mean():.2f}°C')
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Frequency')
plt.title('Distribution of Prophet Prediction Errors')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('visualizations/15_prophet_error_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/15_prophet_error_distribution.png")
plt.show()

# ============================================================================
# STEP 7: VISUALIZE COMPONENTS
# ============================================================================

print("\n" + "="*70)
print("ANALYZING SEASONALITY COMPONENTS")
print("="*70)

print("\nGenerating component plots...")

# Prophet has built-in component plotting
fig = model.plot_components(forecast)
plt.tight_layout()
plt.savefig('visualizations/16_prophet_components.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/16_prophet_components.png")
plt.show()

print("\nComponent Analysis:")
print("  - Trend: Overall temperature changes over time")
print("  - Yearly: Seasonal pattern (summer/winter)")
print("  - Weekly: Day-of-week effects")
print("  - Daily: Hour-of-day effects (day/night cycle)")

# ============================================================================
# STEP 8: FORECAST FUTURE (NEXT 7 DAYS)
# ============================================================================

print("\n" + "="*70)
print("FORECASTING NEXT 7 DAYS")
print("="*70)

print("\nRetraining model on full dataset.")

# Retrain on full dataset
full_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    interval_width=0.95
)
full_model.fit(prophet_df)

# Create future dataframe for next 7 days (168 hours)
future_dates = full_model.make_future_dataframe(periods=168, freq='H')
future_forecast = full_model.predict(future_dates)

# Get only the future predictions
future_only = future_forecast[future_forecast['ds'] > prophet_df['ds'].max()].copy()

print(f"✅ Future forecast generated!")
print(f"\nForecast period: {future_only['ds'].min()} to {future_only['ds'].max()}")

# Plot future forecast
plt.figure(figsize=(15, 6))
plt.plot(prophet_df['ds'].tail(168), prophet_df['y'].tail(168), 
         label='Historical (Last 7 Days)', color='blue', linewidth=1)
plt.plot(future_only['ds'], future_only['yhat'], 
         label='Forecast (Next 7 Days)', color='red', linewidth=2, linestyle='--')
plt.fill_between(future_only['ds'], future_only['yhat_lower'], future_only['yhat_upper'],
                 alpha=0.3, color='red', label='95% Confidence')
plt.axvline(x=prophet_df['ds'].max(), color='green', linestyle=':', 
            linewidth=2, label='Forecast Start', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Prophet: 7-Day Temperature Forecast with Confidence Intervals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/17_prophet_future_forecast.png', dpi=300, bbox_inches='tight')
print("✅ Saved: visualizations/17_prophet_future_forecast.png")
plt.show()

# Print daily summary
print("\n7-Day Forecast Summary:")
daily_summary = future_only.set_index('ds').resample('D').agg({
    'yhat': 'mean',
    'yhat_lower': 'min',
    'yhat_upper': 'max'
})

for date, row in daily_summary.iterrows():
    print(f"  {date.strftime('%A, %Y-%m-%d')}: {row['yhat_lower']:.1f}°C - {row['yhat_upper']:.1f}°C (avg: {row['yhat']:.1f}°C)")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save predictions
results.to_csv('data/processed/prophet_predictions.csv', index=False)
print("✅ Saved predictions: data/processed/prophet_predictions.csv")

# Save future forecast
future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    'data/processed/prophet_future_forecast_7days.csv', index=False
)
print("✅ Saved future forecast: data/processed/prophet_future_forecast_7days.csv")

# ============================================================================
# STEP 10: MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON: SARIMA vs PROPHET")
print("="*70)

# Load SARIMA results for comparison
try:
    sarima_results = pd.read_csv('data/processed/sarima_predictions.csv')
    sarima_results['date'] = pd.to_datetime(sarima_results['date'])
    
    # Calculate SARIMA metrics
    sarima_mae = mean_absolute_error(sarima_results['actual'], sarima_results['predicted'])
    sarima_rmse = np.sqrt(mean_squared_error(sarima_results['actual'], sarima_results['predicted']))
    
    print(f"\nPerformance Comparison:")
    print(f"\n  SARIMA Model:")
    print(f"    MAE:  {sarima_mae:.2f}°C")
    print(f"    RMSE: {sarima_rmse:.2f}°C")
    print(f"\n  Prophet Model:")
    print(f"    MAE:  {mae:.2f}°C")
    print(f"    RMSE: {rmse:.2f}°C")
    
    # Determine winner
    print(f"\nWinner:")
    if mae < sarima_mae:
        improvement = ((sarima_mae - mae) / sarima_mae) * 100
        print(f"  Prophet is better by {improvement:.1f}%!")
        print(f"  Prophet reduces error by {sarima_mae - mae:.2f}°C")
    elif mae > sarima_mae:
        improvement = ((mae - sarima_mae) / mae) * 100
        print(f"  SARIMA is better by {improvement:.1f}%!")
        print(f"  SARIMA reduces error by {mae - sarima_mae:.2f}°C")
    else:
        print(f"  It's a tie! Both models perform equally well.")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model Comparison: SARIMA vs Prophet', fontsize=16, fontweight='bold')
    
    # MAE comparison
    models = ['SARIMA', 'Prophet']
    mae_values = [sarima_mae, mae]
    axes[0].bar(models, mae_values, color=['steelblue', 'coral'], alpha=0.7)
    axes[0].set_ylabel('MAE (°C)')
    axes[0].set_title('Mean Absolute Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.05, f'{v:.2f}°C', ha='center', fontweight='bold')
    
    # RMSE comparison
    rmse_values = [sarima_rmse, rmse]
    axes[1].bar(models, rmse_values, color=['steelblue', 'coral'], alpha=0.7)
    axes[1].set_ylabel('RMSE (°C)')
    axes[1].set_title('Root Mean Squared Error')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.05, f'{v:.2f}°C', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/18_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: visualizations/18_model_comparison.png")
    plt.show()
    
except FileNotFoundError:
    print("\nSARIMA results not found. Run SARIMA model first for comparison.")

print("\n" + "="*70)
print("PROPHET MODEL COMPLETE!")
print("="*70)

print(f"\nSummary:")
print(f"  Model: Prophet with automatic seasonality detection")
print(f"  Accuracy: MAE = {mae:.2f}°C, RMSE = {rmse:.2f}°C")
print(f"  Visualizations: 8 new charts created")
print(f"  Future forecast: Next 7 days with confidence intervals")

