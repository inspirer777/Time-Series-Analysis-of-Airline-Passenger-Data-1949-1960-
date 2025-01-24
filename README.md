# Time Series Analysis of Airline Passengers Dataset

This project demonstrates time series analysis on the "AirPassengers" dataset, which contains monthly airline passenger numbers from 1949 to 1960. The analysis includes various techniques like seasonal decomposition, stationarity tests, and filtering to extract meaningful insights from the data.

## Libraries Used:
- `pandas` for data manipulation
- `numpy` for numerical calculations
- `matplotlib` for plotting graphs
- `statsmodels` for statistical models and time series analysis
- `hvplot` for interactive plotting
- `pathlib` for working with file paths

## Steps in the Code:
1. **Loading the Dataset**: The dataset is fetched from `statsmodels` using `get_rdataset("AirPassengers")` and preprocessed.
2. **Plotting Data**: We plot the monthly airline passenger numbers from 1949 to 1960.
3. **Seasonal Decomposition**: The dataset is decomposed into trend, seasonal, and residual components using the `seasonal_decompose` method.
4. **Stationarity Tests**: The Augmented Dickey-Fuller (ADF) and KPSS tests are applied to check the stationarity of the data.
5. **Filtering**: The Hodrick-Prescott (HP) filter is applied to extract the cyclical and trend components of the time series.
6. **Differencing & Transformation**: The data is differenced and log-transformed to make the series stationary.

## How to Run:
1. Ensure you have all the necessary libraries installed:
   ```bash
   pip install pandas numpy matplotlib statsmodels hvplot
   ```

## Good lucl ! 
