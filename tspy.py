import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.datasets import get_rdataset
import statsmodels
from statsmodels.tsa.filters.hp_filter import hpfilter
air_passengers = get_rdataset("AirPassengers")
airp_df = air_passengers.data
airp_df.index = pd.date_range('1949', '1961', freq='M')
airp_df.drop(columns=['time'], inplace=True)
print(airp_df.index)
print(airp_df)
import matplotlib.pyplot as plt
import hvplot as hv
airp_df['value'].plot(
  title='monthly airline passenger numbers 1949-1960'
);plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose, STL
air_decomposed = seasonal_decompose(airp_df,model='multiplicative')
air_decomposed.plot(); plt.show()
air_decomposed.trend.plot(); plt.show()
air_decomposed.seasonal.plot(); plt.show()
air_decomposed.resid.plot(); plt.show()
(air_decomposed.trend * air_decomposed.seasonal * air_decomposed.resid).plot();plt.show()
from statsmodels.tsa.stattools import adfuller, kpss
def print_results(output, test='adf'):
  pval = output[1]test_score = output[0]
lags = output[2]
decision = 'Non-Stationary'
if test == 'adf':
  critical = output[4]
if pval <= 0.05:
  decision = 'Stationary'
elif test=='kpss':
  critical = output[3]
if pval > 0.05:
  decision = 'Stationary'
output_dict = {
  'Test Statistic': test_score,
  'p-value': pval,
  'Numbers of lags': lags,
  'decision': decision
}
for key, value in critical.items():
  output_dict["Critical Value (%s)" % key] = value
return pd.Series(output_dict, name=test)
adf_output = adfuller(airp_df)
kpss_output = kpss(airp_df)
print(adf_output)
print(kpss_output)
def check_stationarity(df):
  kps = kpss(df)adf = adfuller(df)
kpss_pv, adf_pv = kps[1], adf[1]
kpssh, adfh = 'Stationary', 'Non-stationary'
if adf_pv < 0.05:
  # Reject ADF Null Hypothesis
  adfh = 'Stationary'
if kpss_pv < 0.05:
  # Reject KPSS Null Hypothesis
  kpssh = 'Non-stationary'
return (kpssh, adfh)
print([check_stationarity(airp_df),check_stationarity(airp_df)])
def plot_comparison(methods, plot_type='line'):
  n = len(methods) // 2
fig, ax = plt.subplots(n,2, sharex=True, figsize=(20,10))
for i, method in enumerate(methods):
  method.dropna(inplace=True)
name = [n for n in globals() if globals()[n] is method]
v, r = i // 2, i % 2
kpss_s, adf_s = check_stationarity(method)
method.plot(kind=plot_type,
            ax=ax[v,r],
            legend=False,
            title=f'{name[0]} --> KPSS: {kpss_s}, ADF{adf_s}')
ax[v,r].title.set_size(20)
method.rolling(52).mean().plot(ax=ax[v,r],legend=False)
from statsmodels.tsa.filters.hp_filter import hpfilter
plt.rcParams["figure.figsize"] = (20,3)
airp_cyclic, airp_trend = hpfilter(airp_df)
fig, ax = plt.subplots(1,2)
airp_cyclic.plot(ax=ax[0], title='airp Cyclic Component')
airp_trend.plot(ax=ax[1], title='airp Trend Component')
ax[0].title.set_size(20); ax[1].title.set_size(20)
first_order_diff = airp_df.diff().dropna()
differencing_twice = airp_df.diff(52).diff().dropna()
rolling = airp_df.rolling(window=52).mean()
subtract_rolling_mean = airp_df - rolling
log_transform = np.log(airp_df)
decomp = seasonal_decompose(airp_df)
sd_detrend = decomp.observed - decomp.trend
cyclic, trend = hpfilter(airp_df)
fig, ax = plt.subplots(1,2)
airp_cyclic.plot(ax=ax[0], title='airp Cyclic Component')
airp_trend.plot(ax=ax[1], title='airp Trend Component')
ax[0].title.set_size(20); ax[1].title.set_size(20)
