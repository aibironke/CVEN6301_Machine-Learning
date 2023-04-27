library(tidyverse)
library(lubridate)
library(zoo)
library(xts)
library(tseries)

#Setting working directory
path <- 'D:/CVEN6301_Machine_Learning/Timeseries'
setwd(path)

#Load the CSV data into a data frame
j17 <- read.csv('J-17.csv')

#Convert date column to Date format
j17$DailyHighDate <- as.Date(j17$DailyHighDate)

#Check for missing values
missing_values <- sum(is.na(j17$WaterLevelElevation))
cat("Number of missing values in the time series data:", missing_values, "\n")

#Remove duplicate rows
j17 <- distinct(j17)


#Identify and replace outliers with NA
outliers <- j17$WaterLevelElevation > 703.2 | j17$WaterLevelElevation < 612.5
j17$WaterLevelElevation[outliers] <- NA

#Remove rows with missing values
j17 <- j17 %>% drop_na()

#Plot the time series data
ggplot(j17, aes(x = DailyHighDate, y = WaterLevelElevation)) +
  geom_line() +
  ggtitle("J17 Annual Time Series") +
  labs(x = "DailyHighDate", y = "WaterLevelElevation")

#Perform autocorrelation test
acf_data <- acf(j17$WaterLevelElevation, lag.max = 20, plot = TRUE)
summary(acf_data)

#Create a zoo object
j17_zoo <- zoo(j17$WaterLevelElevation, j17$DailyHighDate)

#Interpolate the missing values using na.approx
j17_zoo_interp <- na.approx(j17_zoo)

#Plot the interpolated time series data
autoplot(j17_zoo_interp) +
  ggtitle("J17 Interpolated Time Series") +
  labs(x = "Year", y = "Water Level")

#Test for Stationarity
adf.test(j17_zoo_interp)

#Calculate the 10-day moving average
ts_j17 <- ts(j17$WaterLevelElevation, start = c(1932, 11), end = c(2023, 3), frequency = 365)
ma_10d <- rollmean(ts_j17, k = 10, na.rm = TRUE, align = "right")

#Plot the moving average data
autoplot(ma_10d) +
  ggtitle("J17 10-day Moving Average Time Series") +
  labs(x = "Year", y = "Water Level")

# Perform seasonal decomposition
j17_decomp <- decompose(ma_10d)

# Extract the components
trend <- j17_decomp$trend
seasonal <- j17_decomp$seasonal
random <- j17_decomp$random

# Plot the original time series data and the trend, seasonal, and residual components of the decomposition
autoplot(trend) +
  ggtitle("J17 trend Time Series") +
  labs(x = "Year", y = "Water Level")

autoplot(seasonal) +
  ggtitle("J17 Seasonal Time Series") +
  labs(x = "Year", y = "Water Level")

autoplot(random) +
  ggtitle("J17 Random Time Series") +
  labs(x = "Year", y = "Water Level")
