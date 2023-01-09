# Here's a general outline of the steps you can follow:

# Import the necessary libraries and download the daily stock data using yfinance.
# Visualize the time series data to get a sense of the trend and seasonality.
# Decompose the time series data into trend, seasonality, and residuals using 
# the seasonal decompose function from statsmodels.
# Use the ACF and PACF plots to identify the appropriate values for the p, d, 
# and q parameters in the SARIMA model.
# Use grid search to find the optimal values for the seasonal p, d, and q 
# parameters, as well as the periodicity of the seasonality.
# Train the SARIMA model using the optimal parameters and use it to make 
# predictions for the next 3 days.


# Here's some code that demonstrates these steps:
    

#%%

import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

from sklearn.metrics import mean_absolute_error



#%%

# Fetch the stock data
ticker = yf.Ticker('XU030.IS')
data = ticker.history(period="1y")

# Preprocess the data
data.dropna(inplace=True)
data.sort_index(inplace=True)

# Split the data into training and test sets
train = data[:-3]
test = data[-4:]
test= test['Close']

train = train['Close']


# Visualize the time series data
plt.plot(train)
plt.show()


#%%

# Decomposing a time series data means breaking it down into its individual components. 
# When you decompose a time series data, you are typically trying to identify and separate 
# out the trend, seasonality, and residuals of the data.

# The trend is the long-term direction of the data, and represents the underlying pattern 
# of the data over time. It can be increasing, decreasing, or relatively constant.

# The seasonality is the periodic fluctuations in the data that occur at regular intervals. 
# For example, a retail store might see an increase in sales during the holiday season, 
# or a temperature dataset might show higher temperatures in the summer and lower temperatures 
# in the winter.

# The residuals are the residuals left after removing the trend and seasonality from the data. 
# They represent the short-term fluctuations in the data that cannot be explained by the trend 
# and seasonality.

# Decomposing a time series data can be useful for a variety of purposes, including understanding 
# the underlying patterns in the data, identifying outliers, and improving the accuracy of forecasts.





# Use grid search to find the optimal values for the seasonal p, d, and q parameters, 
# as well as the periodicity of the seasonality also m=5 because our data's frequency is 5 days.
# Because stock markets works only during working days from monday till friday.




# Decompose the time series data into trend, seasonality, and residuals
decomposition = seasonal_decompose(train, period=5)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.plot(trend)
plt.show()

plt.plot(seasonal)
plt.show()

plt.plot(residual)
plt.show()


# In the seasonal_decompose function from Python's statsmodels library, the period parameter 
# specifies the number of observations per seasonal cycle. For example, if you are decomposing 
# a monthly time series and the seasonal pattern repeats every year, you would set period=12. 
# If you are decomposing a daily time series and the seasonal pattern repeats every week, you 
# would set period=7.

# The value you choose for period should be consistent with the frequency of the time series data. 
# For example, if you are working with a quarterly time series, you would set period=4, and if you 
# are working with a yearly time series, you would set period=1.

# It's important to choose the correct value for period because it affects the way the seasonal 
# component of the time series is calculated. If you set period to the wrong value, the decomposition 
# may not accurately capture the seasonal pattern in the data.


#%%

# Plot the ACF and PACF to identify the appropriate p, d, and q values
plot_acf(train)
plot_pacf(train)
plt.show()

#%%

# differential is taken to get rid of autocorrelation.
# Calculate the difference between consecutive elements
df_diff = train.diff()

plt.plot(df_diff)
plt.show()

# Plot the ACF and PACF to identify the appropriate p, d, and q values
plot_acf(df_diff)
plot_pacf(df_diff)
plt.show()


#%%

# Decompose the df_diff time series data into trend, seasonality, and residuals
decomposition = seasonal_decompose(df_diff, period=5)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.plot(trend)
plt.show()

plt.plot(seasonal)
plt.show()

plt.plot(residual)
plt.show()



#%%

# Start from the second value because of the first value is NaN.
df_diff.dropna(inplace=True)

model = auto_arima(df_diff, seasonal=True, m=5)

#%%

# Train the SARIMA model and use it to make predictions for the next 3 days
model.fit(df_diff)
predictions = model.predict(n_periods=3)
print(predictions)


#%%

# take the difference of the test data
test_diff = test.diff()
test_diff.dropna(inplace = True)

# Calculate the mean absolute error
mae = mean_absolute_error(test_diff, predictions)
print(mae)


#%%

# Lets print every important infos

print(predictions)
print(df_diff)
print(test_diff)
print(model)