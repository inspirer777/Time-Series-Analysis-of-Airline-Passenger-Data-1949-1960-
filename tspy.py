# Import necessary libraries
import pandas as pd  # Library for data manipulation
import numpy as np  # Library for numerical calculations
from pathlib import Path  # To work with file paths
from statsmodels.datasets import get_rdataset  # To load datasets from statsmodels
import statsmodels  # For statistical models
from statsmodels.tsa.filters.hp_filter import hpfilter  # For HP filter to separate trend and cyclical components

# Load the "AirPassengers" dataset
air_passengers = get_rdataset("AirPassengers")  # Getting the dataset from statsmodels

# Convert the dataset to a pandas DataFrame
airp_df = air_passengers.data

# Set the index of the DataFrame to a monthly time series from 1949 to 1961
airp_df.index = pd.date_range('1949', '1961', freq='M')

# Drop the 'time' column as it's unnecessary
airp_df.drop(columns=['time'], inplace=True)

# Print the index and the DataFrame to check the data
print(airp_df.index)
print(airp_df)

# Plotting the data using matplotlib to visualize the passenger numbers
import matplotlib.pyplot as plt  # For plotting graphs
import hvplot as hv  # For interactive plotting
airp_df['value'].plot(
    title='Monthly Airline Passenger Numbers 1949-1960'
)
plt.show()

# Perform seasonal decomposition using the multiplicative model
from statsmodels.tsa.seasonal import seasonal_decompose, STL  # For seasonal decomposition
air_decomposed = seasonal_decompose(airp_df, model='multiplicative')

# Plot the seasonal decomposition results
air_decomposed.plot()
plt.show()

# Plot individual components of the decomposition (trend, seasonal, residual)
air_decomposed.trend.plot()
plt.show()

air_decomposed.seasonal.plot()
plt.show()

air_decomposed.resid.plot()
plt.show()

# Combine trend, seasonal, and residual components and plot
(air_decomposed.trend * air_decomposed.seasonal * air_decomposed.resid).plot()
plt.show()

# Perform stationarity tests (ADF and KPSS)
from statsmodels.tsa.stattools import adfuller, kpss  # For stationarity tests

# Define a function to print test results
def print_results(output, test='adf'):
    pval = output[1]
    test_score = output[0]
    lags = output[2]
    decision = 'Non-Stationary'
    
    # Check the result of the test and make a decision
    if test == 'adf':
        critical = output[4]
        if pval <= 0.05:
            decision = 'Stationary'
    elif test == 'kpss':
        critical = output[3]
        if pval > 0.05:
            decision = 'Stationary'
    
    output_dict = {
        'Test Statistic': test_score,
        'p-value': pval,
        'Number of lags': lags,
        'decision': decision
    }
    
    for key, value in critical.items():
        output_dict[f"Critical Value ({key})"] = value
    
    return pd.Series(output_dict, name=test)

# Run ADF and KPSS tests
adf_output = adfuller(airp_df)
kpss_output = kpss(airp_df)

# Print results of the tests
print(adf_output)
print(kpss_output)

# Function to check stationarity of the series using both ADF and KPSS tests
def check_stationarity(df):
    kps = kpss(df)
    adf = adfuller(df)
    
    kpss_pv, adf_pv = kps[1], adf[1]
    kpssh, adfh = 'Stationary', 'Non-stationary'
    
    if adf_pv < 0.05:
        adfh = 'Stationary'
    
    if kpss_pv < 0.05:
        kpssh = 'Non-stationary'
    
    return (kpssh, adfh)

# Check and print stationarity status of the data
print([check_stationarity(airp_df), check_stationarity(airp_df)])

# Function to plot comparison of different methods
def plot_comparison(methods, plot_type='line'):
    n = len(methods) // 2
    fig, ax = plt.subplots(n, 2, sharex=True, figsize=(20, 10))
    
    for i, method in enumerate(methods):
        method.dropna(inplace=True)
        name = [n for n in globals() if globals()[n] is method]
        v, r = i // 2, i % 2
        kpss_s, adf_s = check_stationarity(method)
        method.plot(kind=plot_type, ax=ax[v, r], legend=False,
                    title=f'{name[0]} --> KPSS: {kpss_s}, ADF: {adf_s}')
        ax[v, r].title.set_size(20)
        method.rolling(52).mean().plot(ax=ax[v, r], legend=False)

# Apply HP Filter to extract cyclic and trend components
plt.rcParams["figure.figsize"] = (20, 3)
airp_cyclic, airp_trend = hpfilter(airp_df)

# Plot the cyclic and trend components of the series
fig, ax = plt.subplots(1, 2)
airp_cyclic.plot(ax=ax[0], title='Airp Cyclic Component')
airp_trend.plot(ax=ax[1], title='Airp Trend Component')
ax[0].title.set_size(20)
ax[1].title.set_size(20)

# Apply differencing and log transformation to make the series stationary
first_order_diff = airp_df.diff().dropna()
differencing_twice = airp_df.diff(52).diff().dropna()
rolling = airp_df.rolling(window=52).mean()
subtract_rolling_mean = airp_df - rolling
log_transform = np.log(airp_df)

# Perform seasonal decomposition and plot the detrended series
decomp = seasonal_decompose(airp_df)
sd_detrend = decomp.observed - decomp.trend

# Apply HP filter again to see the cyclic and trend components
cyclic, trend = hpfilter(airp_df)
fig, ax = plt.subplots(1, 2)
airp_cyclic.plot(ax=ax[0], title='Airp Cyclic Component')
airp_trend.plot(ax=ax[1], title='Airp Trend Component')
ax[0].title.set_size(20)
ax[1].title.set_size(20)
