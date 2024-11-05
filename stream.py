import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from dateutil.relativedelta import relativedelta

# Function to take and validate user input for months, dates, and values
def get_user_data():
    # Get the total number of months from the user, starting with a blank input field
    total_months = st.number_input("Enter the total number of months:", min_value=1, step=1, value=None)

    # Get the start date from the user, initially blank
    start_date = st.date_input("Enter the start date:")

    # Check if both inputs are provided before proceeding
    if total_months and start_date:
        # Calculate the end date automatically based on total months
        end_date = pd.to_datetime(start_date) + relativedelta(months=total_months - 1)

        # Generate dates between start and end date
        months = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Get values from the user as a list
        values_input = st.text_area(f"Enter {total_months} values separated by commas:")

        # Convert input string to a list of floats, ignoring spaces around commas
        values = [float(v.strip()) for v in values_input.split(',') if v.strip()]

        if len(values) != total_months:
            st.error(f"The number of values entered does not match the total number of months ({total_months}).")
            return None

        return pd.DataFrame({'Month': months, 'Value': values}).set_index('Month')
    return None

# Load user data
data = get_user_data()
if data is None:
    st.stop()

# Compute the autocorrelation function (ACF)
lag_acf = acf(data['Value'], nlags=24)

# Function to detect seasonality or outliers dynamically
def detect_seasonality_or_outlier(acf_values, data, threshold=0.2, min_lag=2, z_threshold=2.0):
    peaks = [i for i in range(min_lag, len(acf_values)) if acf_values[i] > threshold]
    if not peaks:
        return None, None

    peak_acf = max(peaks, key=lambda x: acf_values[x])

    # Group data by month and calculate mean
    monthly_means = data.groupby(data.index.month)['Value'].mean()

    # Calculate z-scores for monthly means to identify outliers
    monthly_zscores = zscore(monthly_means)

    # Check if any month is an outlier based on z-score threshold
    outlier_month = monthly_means.index[np.argmax(monthly_zscores)]
    if monthly_zscores.max() > z_threshold:
        # Return the month name instead of number
        return peak_acf, f"outlier in {pd.Timestamp(2024, outlier_month, 1).strftime('%B')}"

    # If no outliers, determine seasonality pattern
    seasonal_values = data['Value'][::peak_acf]
    if len(seasonal_values) < 2:
        return peak_acf, None

    seasonal_trend = "increase" if seasonal_values.diff().mean() > 0 else "decrease"
    return peak_acf, seasonal_trend

# Detect seasonality or outlier dynamically
seasonal_period, seasonal_trend = detect_seasonality_or_outlier(lag_acf, data)

# Function to calculate 3-month moving average of slopes for a given year
def moving_average_slopes(data, year, window=3):
    # Filter the data for the specified year
    data_year = data[data.index.year == year]
   
    # Calculate the slopes for each consecutive period in the selected year
    slopes = np.diff(data_year['Value'])
   
    # Calculate the moving average of the slopes
    ma_slopes = np.convolve(slopes, np.ones(window) / window, mode='valid')
   
    # Check if the slope of the moving average approaches zero
    if np.abs(ma_slopes.mean()) < 0.0001:
        ma_slope_trend = "The trend has stabilized since the beginning of the year"
    elif ma_slopes.mean() > 0:
        ma_slope_trend = "The trend has increased since the beginning of the year"
    else:
        ma_slope_trend = "The trend has decreased since the beginning of the year"
   
    return ma_slopes, ma_slope_trend

# Trend analysis and summary function
def trend_analysis(data, seasonal_period, seasonal_trend):
    x = np.arange(len(data))
    y = data['Value'].values
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

    # Calculate total change from start to end
    total_change = y[-1] - y[0]
   
    slopes = [data['Value'].iloc[i] - data['Value'].iloc[i-1] for i in range(1, len(data))]
    max_increase = max(slopes)
    max_decrease = min(slopes)
    max_increase_periods = [(data.index[i].strftime('%B %Y'), data.index[i+1].strftime('%B %Y')) for i, slope in enumerate(slopes) if slope == max_increase]
    max_decrease_periods = [(data.index[i].strftime('%B %Y'), data.index[i+1].strftime('%B %Y')) for i, slope in enumerate(slopes) if slope == max_decrease]

    max_increase_periods_str = ', '.join([f"{start} to {end}" for start, end in max_increase_periods])
    max_decrease_periods_str = ', '.join([f"{start} to {end}" for start, end in max_decrease_periods])

    # Moving average calculation
    ma_window = 3  # 3-month moving average
    moving_avg = np.convolve(data['Value'].values, np.ones(ma_window) / ma_window, mode='valid')
    moving_avg_trend = (moving_avg[-1] - moving_avg[0]) / (len(moving_avg) - 1)
    ma_trend_direction = "upward" if moving_avg_trend > 0 else "downward" if moving_avg_trend < 0 else "stable"

    # Additional moving average slope analysis for the latest year dynamically
    latest_year = data.index.year.max()
    ma_slopes_latest, ma_slope_trend_latest = moving_average_slopes(data, year=latest_year)

    # Summary
    if seasonal_period:
        if seasonal_trend.startswith('outlier'):
            seasonal_summary = f"An outlier has been detected in {seasonal_trend.split()[-1]} with a significant pattern repeating every {seasonal_period} months."
        else:
            seasonal_summary = f"A seasonal pattern repeating every {seasonal_period} months has been detected, with a general {seasonal_trend} in trend."
    else:
        seasonal_summary = "No seasonal pattern was detected, indicating an outlier."

    summary = (
        f"\n\n****SUMMARY****\nBy examining these insights, we can conclude that the overall value trend is "
        f"{direction} by {total_change:.4f} from start to end. The 3-month moving average indicates a {ma_trend_direction} trend, highlighting the general direction "
        f"of the data.\n\nSignificant fluctuations are evident, with the largest increase observed between "
        f"{max_increase_periods_str} (slope: {max_increase:.4f}) and the largest "
        f"decrease from {max_decrease_periods_str} (slope: {max_decrease:.4f})."
        f"\n\nAdditionally, {seasonal_summary}\n\n"
        f"**Trend for {latest_year}:** {ma_slope_trend_latest}."
    )

    return summary

# Get the summary report
summary = trend_analysis(data, seasonal_period, seasonal_trend)
st.text(summary)

# Define ma_window before plotting
ma_window = 3  # 3-month moving average
moving_avg = np.convolve(data['Value'].values, np.ones(ma_window) / ma_window, mode='valid')

# Plot the data and moving average
st.subheader("Data Plot with 3-Month Moving Average")
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Value'], label='Original Data', marker='o')
plt.plot(data.index[ma_window-1:], moving_avg, label='3-Month Moving Average', linestyle='--', color='orange')
plt.ylabel('Values')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(plt)
