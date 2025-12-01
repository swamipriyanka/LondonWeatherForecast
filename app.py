"""
London Weather Forecasting Dashboard
Interactive Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="London Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #262730 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #262730 !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load the weather data"""
    df = pd.read_csv('data/processed/london_weather_with_features.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df

@st.cache_data
def load_predictions():
    """Load SARIMA predictions"""
    try:
        sarima_pred = pd.read_csv('data/processed/sarima_predictions.csv')
        sarima_pred['date'] = pd.to_datetime(sarima_pred['date'])
        return sarima_pred
    except:
        return None

def train_and_forecast(data, forecast_days):
    """Train SARIMA model and generate forecast"""
    
    # Resample to daily
    daily_temp = data['temperature'].resample('D').mean()
    
    # Train model
    model = SARIMAX(
        daily_temp,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    with st.spinner('Training model... This may take a moment...'):
        fitted_model = model.fit(disp=False, maxiter=100, method='nm')
    
    # Forecast
    forecast = fitted_model.forecast(steps=forecast_days)
    
    return forecast, fitted_model

# ============================================================================
# LOAD DATA
# ============================================================================

try:
    df = load_data()
    sarima_predictions = load_predictions()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🌤️ Weather Forecast")
st.sidebar.markdown("---")

# Forecast settings
st.sidebar.subheader("Forecast Settings")
forecast_days = st.sidebar.slider(
    "Forecast Period (days)",
    min_value=1,
    max_value=30,
    value=7,
    help="Select how many days to forecast"
)

# Data info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
**Location:** London, UK  
**Period:** {df.index.min().date()} to {df.index.max().date()}  
**Total Records:** {len(df):,} hours  
**Features:** Temperature, Humidity, Precipitation, Wind, Pressure
""")

# Model info
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Info")
if sarima_predictions is not None:
    mae = np.mean(np.abs(sarima_predictions['actual'] - sarima_predictions['predicted']))
    st.sidebar.success(f"""
**Model:** SARIMA(1,0,1)x(1,1,0,7)  
**Accuracy (MAE):** {mae:.2f}°C  
**Status:** ✅ Excellent
""")
else:
    st.sidebar.warning("Predictions not available")

# About
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
This dashboard forecasts London's temperature using time series analysis.

**Technologies:**
- Python
- SARIMA Model
- Streamlit
- Plotly

**Created by:** Priyanka  
**GitHub:** https://github.com/swamipriyanka/LondonWeatherForecast
""")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Title
st.title("🌦️ London Weather Forecasting Dashboard")
st.markdown("### Real-time Temperature Predictions using SARIMA Model")
st.markdown("---")

# Current weather metrics
col1, col2, col3, col4 = st.columns(4)

latest_data = df.iloc[-1]
with col1:
    st.metric(
        label="🌡️ Current Temperature",
        value=f"{latest_data['temperature']:.1f}°C",
        delta=f"{latest_data['temperature'] - df['temperature'].iloc[-25]:.1f}°C vs yesterday"
    )

with col2:
    st.metric(
        label="💧 Humidity",
        value=f"{latest_data['humidity']:.0f}%"
    )

with col3:
    st.metric(
        label="💨 Wind Speed",
        value=f"{latest_data['wind_speed']:.1f} km/h"
    )

with col4:
    st.metric(
        label="🌡️ Pressure",
        value=f"{latest_data['pressure']:.0f} hPa"
    )

st.markdown("---")

# ============================================================================
# HISTORICAL DATA VISUALIZATION
# ============================================================================

st.subheader("📈 Historical Temperature Trends")

# Time range selector
time_range = st.radio(
    "Select time range:",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
    horizontal=True
)

# Filter data based on selection
if time_range == "Last 7 Days":
    plot_df = df.last('7D')
elif time_range == "Last 30 Days":
    plot_df = df.last('30D')
elif time_range == "Last 90 Days":
    plot_df = df.last('90D')
else:
    plot_df = df

# Resample to daily for cleaner visualization
daily_df = plot_df['temperature'].resample('D').agg(['mean', 'min', 'max'])

# Create interactive plot
fig = go.Figure()

# Add mean temperature
fig.add_trace(go.Scatter(
    x=daily_df.index,
    y=daily_df['mean'],
    mode='lines',
    name='Average',
    line=dict(color='rgb(31, 119, 180)', width=2),
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Avg Temp</b>: %{y:.1f}°C<extra></extra>'
))

# Add min/max range
fig.add_trace(go.Scatter(
    x=daily_df.index,
    y=daily_df['max'],
    mode='lines',
    name='Max',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=daily_df.index,
    y=daily_df['min'],
    mode='lines',
    name='Min/Max Range',
    fill='tonexty',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(width=0),
    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Range</b>: %{y:.1f}°C<extra></extra>'
))

fig.update_layout(
    title="Daily Temperature with Min/Max Range",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# GENERATE FORECAST
# ============================================================================

st.subheader(f"🔮 {forecast_days}-Day Temperature Forecast")

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        st.session_state.generate_forecast = True

if 'generate_forecast' not in st.session_state:
    st.session_state.generate_forecast = False

if st.session_state.generate_forecast or forecast_days:
    
    # Generate forecast
    forecast, model = train_and_forecast(df, forecast_days)
    
    # Create forecast dataframe
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'temperature': forecast.values
    })
    
    # Display forecast
    st.success(f"✅ Forecast generated for {forecast_days} days ahead!")
    
    # Forecast visualization
    fig_forecast = go.Figure()
    
    # Historical data (last 30 days)
    hist_daily = df['temperature'].resample('D').mean().tail(30)
    fig_forecast.add_trace(go.Scatter(
        x=hist_daily.index,
        y=hist_daily.values,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Temp</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['temperature'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    # Add vertical line at forecast start
    fig_forecast.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dot",
        line_color="green",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig_forecast.update_layout(
        title=f"{forecast_days}-Day Temperature Forecast",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        height=450
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast table
    st.subheader("📋 Detailed Forecast")
    
    # Format forecast table
    forecast_display = forecast_df.copy()
    forecast_display['Day'] = forecast_display['date'].dt.strftime('%A')
    forecast_display['Date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
    forecast_display['Temperature'] = forecast_display['temperature'].apply(lambda x: f"{x:.1f}°C")
    forecast_display = forecast_display[['Day', 'Date', 'Temperature']]
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
    
    # Download button
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"london_weather_forecast_{forecast_days}days.csv",
        mime="text/csv"
    )

st.markdown("---")

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================

if sarima_predictions is not None:
    st.subheader("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics
        mae = np.mean(np.abs(sarima_predictions['actual'] - sarima_predictions['predicted']))
        rmse = np.sqrt(np.mean((sarima_predictions['actual'] - sarima_predictions['predicted'])**2))
        
        st.markdown(f"""
        **Performance Metrics:**
        - **MAE (Mean Absolute Error):** {mae:.2f}°C
        - **RMSE (Root Mean Squared Error):** {rmse:.2f}°C
        - **Test Period:** {len(sarima_predictions)} days
        
        ✅ The model predicts temperature with an average error of only **{mae:.2f}°C** - excellent accuracy!
        """)
    
    with col2:
        # Actual vs Predicted scatter
        fig_scatter = px.scatter(
            sarima_predictions,
            x='actual',
            y='predicted',
            title='Actual vs Predicted Temperature',
            labels={'actual': 'Actual (°C)', 'predicted': 'Predicted (°C)'},
            opacity=0.6
        )
        
        # Add perfect prediction line
        min_val = min(sarima_predictions['actual'].min(), sarima_predictions['predicted'].min())
        max_val = max(sarima_predictions['actual'].max(), sarima_predictions['predicted'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================================================================
# ADDITIONAL INSIGHTS
# ============================================================================

st.subheader("📊 Additional Weather Insights")

col1, col2 = st.columns(2)

with col1:
    # Monthly average temperature
    monthly_avg = df.groupby(df.index.month)['temperature'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_monthly = go.Figure(data=[
        go.Bar(x=months, y=monthly_avg.values, marker_color='lightblue')
    ])
    fig_monthly.update_layout(
        title="Average Temperature by Month",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        height=300
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    # Hourly pattern
    hourly_avg = df.groupby(df.index.hour)['temperature'].mean()
    
    fig_hourly = go.Figure(data=[
        go.Scatter(x=hourly_avg.index, y=hourly_avg.values, 
                   mode='lines+markers', line=dict(color='coral', width=2))
    ])
    fig_hourly.update_layout(
        title="Average Temperature by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Temperature (°C)",
        height=300
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built using Streamlit | Data from Open-Meteo API | Model: SARIMA</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)