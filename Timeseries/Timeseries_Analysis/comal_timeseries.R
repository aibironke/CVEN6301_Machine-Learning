library(xts)
library(zoo)
library (lubridate)
library(tseries)
library(forecast)

#Setting working directory 
path <- 'D:/CVEN6301_Machine_Learning/Timeseries'
setwd(path)

# Load the CSV data into a data frame
comal <- read.csv('Comal.csv ',skip = 28, header = TRUE)

#drop row 1, columns 1,2&5
comal<- comal[-1, -c(1, 2, 5)]

# Convert date column to Date format
comal$X.1 <- as.Date(comal$X.1,format = "%m/%d/%Y" )

# Extract the relevant time series data
Datetime <- comal$X.1

# Check for missing values
missing_values <- sum(is.na(Datetime))

# Print the number of missing values
cat("Number of missing values in the time series data:", missing_values, "\n")

#Extract beginning and end dates
bgn <- as.Date(comal$X.1[1])
end <- as.Date(comal$X.1[length(comal$X.1)])

# Convert the character column to numeric
comal$X.2 <- as.numeric(comal$X.2)

#check summary
summary(comal)

#Create a zoo object to be able to interpolate missing values
comal_zoo <- zoo(comal$X.2, comal$X.1)

# Interpolate the missing values using na.approx
comal_zoo_interp <- na.approx(comal_zoo)

#check summary
summary(comal_zoo_interp)

# Print the structure of the zoo object
str(comal_zoo_interp)

# Print the start and end values of the values column
cat("Start value: ", coredata(comal_zoo_interp)[1], "\n")
cat("End value: ", coredata(comal_zoo_interp)[length(coredata(comal_zoo_interp))], "\n")

#Autocorrelation Test
acf(comal_zoo_interp)

########Test for Stationarity
#if p-value < 0.05, then null hypothesis can be rejected, data is stationary.
# Meaning the data has constant mean, varaiance and autocorrrelation.

adf.test(comal_zoo_interp)
#data:  comal_zoo_interp
#Dickey-Fuller = -5.7684, Lag order = 32, p-value = 0.01
#alternative hypothesis: stationary

########Decompose the Timeseries to Visualize Trends, Seasonality & other patterns######

# Convert the zoo object to a data frame
library(ggplot2)
comal_ts <- fortify.zoo(comal_zoo_interp)

# Print the resulting data frame
print(comal_ts)

# Convert the dataframe to a time series object
ts_comal <- ts(comal_ts, start=c(2023-04-16, 1), end=c(1927-12-19, 34818), frequency=12)

# Calculate the 10-day moving average
ma_10d <- rollapply(ts_comal, width = 10, FUN = mean, align = "right")


# Perform seasonal decomposition
comal_decomp <- decompose(ma_10d)

# Extract the components
trend <- comal_decomp$trend
seasonal <- comal_decomp$seasonal
random <- comal_decomp$random

# Plot the original time series data and the trend, seasonal, and residual components of the decomposition
plot(ts_comal, main="Original Comal Data")
plot(trend, main="Comal Trend Component")
plot(seasonal, main="Comal Seasonal Component")
plot(random, main="Comal Residual Component")



###########Forecasting######################################
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
