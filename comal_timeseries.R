library(xts)
library(zoo)
library (lubridate)

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

# Remove duplicate rows
comal <- comal[!duplicated(comal), ]

#Extract beginning and end dates
bgn <- as.Date(comal$X.1[1])
end <- as.Date(comal$X.1[length(comal$X.1)])

#Create a zoo object
comal_zoo <- zoo(comal$X.2, comal$X.1)

# Print the structure of the zoo object
str(comal_zoo)

#Print summary of the zoo object
summary(comal)

# Print the start and end values of the values column
cat("Start value: ", coredata(comal_zoo)[1], "\n")
cat("End value: ", coredata(comal_zoo)[length(coredata(comal_zoo))], "\n")

# Check for missing values in the zoo object
missing_values <- is.na(comal_zoo)

# Print the positions of the missing values
print(which(missing_values))

#Autocorrelation Test
acf(comal)

########Test for Stationarity
library(tseries)
adf.test(comal_zoo)
#data:  comal_zoo
#Dickey-Fuller = -5.5565, Lag order = 32, p-value = 0.01
#alternative hypothesis: stationary

########Decompose the Timeseries######

# Convert the zoo object to a data frame
library(ggplot2)
comal_ts <- fortify.zoo(comal_zoo)

# Print the resulting data frame
print(comal_ts)

# Convert the dataframe to a time series object
ts_comal <- ts(comal_ts, start=c(2023-04-16, 1), end=c(1927-12-19, 34818), frequency=12)

# Calculate the 10-day moving average
ma_10d <- rollapply(ts_comal, width = 10, FUN = mean, align = "right")

# Print the result
print(ma_10d)

# Perform seasonal decomposition
comal_decomp <- decompose(ma_10d)

# Extract the components
trend <- comal_decomp$trend
seasonal <- comal_decomp$seasonal
random <- comal_decomp$random

# Plot the original time series data and the trend, seasonal, and residual components of the decomposition
plot(ts_comal, main="Original Comal Data")
plot(trend, main="Trend Component")
plot(seasonal, main="Seasonal Component")
plot(random, main="Residual Component")

########Statistical Test Analysis######
# Perform ADF test
library(urca)

result <- ur.df(ts_comal, type="trend", lags=12)

# Print the test results
summary(result)

acf(ts_comal)
print(acf)
