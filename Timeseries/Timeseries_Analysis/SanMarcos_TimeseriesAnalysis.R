library(xts)
library(zoo)
library (lubridate)
library(tseries)
library(forecast)

#Setting working directory 
path <- 'D:/CVEN6301_Machine_Learning/Timeseries'
setwd(path)

# Load the CSV data into a data frame
sm <- read.csv('SanMarcos.csv ',skip = 29, header = TRUE)

#drop row 1, columns 1,2&5
sm <- sm[-1, -c(1, 2, 5)]

# Convert date column to Date format
sm$datetime <- as.Date(sm$datetime,format = "%m/%d/%Y" )

# Extract the relevant time series data
Datetime <- sm$datetime

# Check for missing values
missing_values <- sum(is.na(Datetime))

# Print the number of missing values
cat("Number of missing values in the time series data:", missing_values, "\n")

#Extract beginning and end dates
bgn <- as.Date(sm$datetime[1])
end <- as.Date(sm$datetime[length(sm$datetime)])

# Convert the character column to numeric
sm$Discharge <- as.numeric(sm$Discharge)

#check summary
summary(sm)

#Create a zoo object to be able to interpolate missing values
sm_zoo <- zoo(sm$Discharge, sm$datetime)

# Interpolate the missing values using na.approx
sm_zoo_interp <- na.approx(sm_zoo)

#check summary
summary(comal_zoo_interp)

# Print the structure of the zoo object
str(sm_zoo_interp)

# Print the start and end values of the values column
cat("Start value: ", coredata(sm_zoo_interp)[1], "\n")
cat("End value: ", coredata(sm_zoo_interp)[length(coredata(sm_zoo_interp))], "\n")

#Autocorrelation Test
acf(sm_zoo_interp)

########Test for Stationarity
#if p-value < 0.05, then null hypothesis can be rejected, data is stationary.
# Meaning the data has constant mean, varaiance and autocorrrelation.

adf.test(sm_zoo_interp)
#data:  sm_zoo_interp
#Dickey-Fuller = -7.2775, Lag order = 29, p-value = 0.01
#alternative hypothesis: stationary

########Decompose the Timeseries to Visualize Trends, Seasonality & other patterns######

# Convert the zoo object to a data frame
library(ggplot2)
sm_ts <- fortify.zoo(sm_zoo_interp)

# Print the resulting data frame
print(sm_ts)

# Convert the dataframe to a time series object
ts_sm <- ts(sm_ts, start=c(1956-05-27, 1), end=c(2023-04-09, 24424), frequency=365)

# Calculate the 10-day moving average
ma_10d <- rollapply(ts_sm, width = 10, FUN = mean, align = "right")

# Perform seasonal decomposition
sm_decomp <- decompose(ma_10d)

# Extract the components
trend <- sm_decomp$trend
seasonal <- sm_decomp$seasonal
random <- sm_decomp$random

#Plot the moving average data
autoplot(sm_decomp) +
  ggtitle("San Marcos 10-day Moving Average Time Series") +
  labs(x = "Year", y = "Discharge")

# Extract the components
trend <- j17_decomp$trend
seasonal <- j17_decomp$seasonal
random <- j17_decomp$random

# Plot the original time series data and the trend, seasonal, and residual components of the decomposition
autoplot(trend) +
  ggtitle("San Marcos trend ") +
  labs(x = "Year", y = "Discharge")

autoplot(seasonal) +
  ggtitle("San Marcos Seasonal") +
  labs(x = "Year", y = "Discharge")

autoplot(random) +
  ggtitle("San Marcos Random Time Series") +
  labs(x = "Year", y = "Discharge")






#################################Forecasting##############################
# Select data range and convert into time series object for Model fitting
ts_comal2 <- ts(comal_ts$comal_zoo_interp, start=c(2018-04-16, 32992), frequency=12)

# Split data into training and testing sets
train_comal <- window(ts_comal2, end =c(2021-04-16, 34088), frequency=12)
test_comal <- window(ts_comal2, start=c(2021-04-17, 34089), frequency=12)

# Fit an ARIMA model to the stationary time series data
arima_model <- arima(train_comal, order=c(1,1,1))

# Fit an ETS model to the training data
ets_model <- ets(train_comal)

# Generate forecasts for the test data using each model
arima_fcst <- forecast(arima_model, h=length(test_comal))
ets_fcst <- forecast(ets_model, h=length(test_comal))

# Compute the accuracy measures for each model
arima_acc <- accuracy(arima_fcst, test_comal)
ets_acc <- accuracy(ets_fcst, test_comal)

# Compare the accuracy measures for the two models
print(arima_acc)
print(ets_acc)
