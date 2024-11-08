import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from dateutil.relativedelta import relativedelta

# Function to reset inputs to default values
def clear_inputs():
    st.session_state['start_month'] = "Select Month"
    st.session_state['start_year'] = "Select Year"
    st.session_state['end_month'] = "Select Month"
    st.session_state['end_year'] = "Select Year"
    st.session_state['total_months'] = None
    st.session_state['values_input'] = ""

# Function to take and validate user input for months, dates, and values
def get_user_data():
    # Ask user to select input method without a pre-selected option
    selection = st.radio(
        "How would you like to specify the date range?",
        ("Start Date and End Date", "Start Date and Total Months"),
        index=None,  # No pre-selection
        key='date_range_selection',
        on_change=clear_inputs  # Clear inputs when selection changes
    )

    # Display inputs based on selection
    month_options = ["Select Month"] + [pd.to_datetime(f"2000-{x}-01").strftime('%B') for x in range(1, 13)]
    year_options = ["Select Year"] + list(range(2000, 2101))
    
    total_months = None
    months = None

    # Option 1: Start Date and End Date
    if selection == "Start Date and End Date":
        start_month = st.selectbox("Select the start month:", month_options, key="start_month")
        start_year = st.selectbox("Select the start year:", year_options, key="start_year")
        
        end_month = st.selectbox("Select the end month:", month_options, key="end_month")
        end_year = st.selectbox("Select the end year:", year_options, key="end_year")
        
        if (start_month != "Select Month" and start_year != "Select Year" and 
            end_month != "Select Month" and end_year != "Select Year"):
            start_month_num = month_options.index(start_month)
            end_month_num = month_options.index(end_month)
            start_date = pd.to_datetime(f"{start_year}-{start_month_num:02d}-01")
            end_date = pd.to_datetime(f"{end_year}-{end_month_num:02d}-01")
            
            if end_date <= start_date:
                st.error("End date must be after the start date.")
                return None
            
            months = pd.date_range(start=start_date, end=end_date, freq='MS')
            total_months = len(months)

    # Option 2: Start Date and Total Months
    elif selection == "Start Date and Total Months":
        start_month = st.selectbox("Select the start month:", month_options, key="start_month")
        start_year = st.selectbox("Select the start year:", year_options, key="start_year")

        total_months = st.number_input("Enter the total number of months:", min_value=1, step=1, value=None, key="total_months")
        
        if start_month != "Select Month" and start_year != "Select Year" and total_months:
            start_month_num = month_options.index(start_month)
            start_date = pd.to_datetime(f"{start_year}-{start_month_num:02d}-01")
            
            end_date = start_date + relativedelta(months=total_months - 1)
            months = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Check if total_months is defined before proceeding to values input
    if total_months is not None and months is not None:
        values_input = st.text_area(f"Enter {total_months} values separated by commas:", key="values_input")
        if values_input:
            values = [float(v.strip()) for v in values_input.split(',') if v.strip()]
            
            if len(values) != total_months:
                st.error(f"The number of values entered does not match the total number of months ({total_months}).")
                return None

            return pd.DataFrame({'Month': months, 'Value': values}).set_index('Month')

    return None

# Load user data
data = get_user_data()

# Stop further execution if data is None
if data is None:
    print("User data is invalid or incomplete. Please re-run the code with valid inputs.")
    raise SystemExit  # Stops further execution in non-Streamlit environments

# Compute the autocorrelation function (ACF)
lag_acf = acf(data['Value'], nlags=24)

# Function to detect seasonality or outliers
def detect_seasonality_or_outlier(acf_values, data, threshold=0.2, min_lag=2, z_threshold=2.0):
    peaks = [i for i in range(min_lag, len(acf_values)) if acf_values[i] > threshold]
    if not peaks:
        return None, None

    peak_acf = max(peaks, key=lambda x: acf_values[x])

    monthly_means = data.groupby(data.index.month)['Value'].mean()
    monthly_zscores = zscore(monthly_means)
    outlier_month = monthly_means.index[np.argmax(monthly_zscores)]
    if monthly_zscores.max() > z_threshold:
        return peak_acf, f"outlier in {pd.Timestamp(2024, outlier_month, 1).strftime('%B')}"

    seasonal_values = data['Value'][::peak_acf]
    if len(seasonal_values) < 2:
        return peak_acf, None

    seasonal_trend = "increase" if seasonal_values.diff().mean() > 0 else "decrease"
    return peak_acf, seasonal_trend

# Detect seasonality or outliers
seasonal_period, seasonal_trend = detect_seasonality_or_outlier(lag_acf, data)

# Function to calculate 3-month moving average of slopes for a given year
def moving_average_slopes(data, year, window=3):
    data_year = data[data.index.year == year]
    slopes = np.diff(data_year['Value'])
    ma_slopes = np.convolve(slopes, np.ones(window) / window, mode='valid')
   
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

    latest_year = data.index[-1].year
    _, ma_slope_trend_latest = moving_average_slopes(data, latest_year)

    # Seasonal summary
    if seasonal_period:
        if seasonal_trend is None:
            seasonal_summary = f"An outlier was detected with a seasonality repeating every {seasonal_period} months."
        elif seasonal_trend == "outlier":
            seasonal_summary = f"An outlier was detected, with a seasonality pattern every {seasonal_period} months."
        else:
            seasonal_summary = f"A seasonal pattern repeating every {seasonal_period} months has been detected, with a general {seasonal_trend} in trend."
    else:
        seasonal_summary = "No seasonal pattern was detected, indicating an outlier."

    summary = (
        f"\n\n****SUMMARY****\nBy examining these insights, we can conclude that the overall value trend is "
        f"{direction} by {total_change:.4f} from start to end. The 3-month moving average indicates a {ma_trend_direction} trend."
        f"The largest increase occurred from {max_increase_periods_str} (slope: {max_increase:.4f}) and the largest "
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
