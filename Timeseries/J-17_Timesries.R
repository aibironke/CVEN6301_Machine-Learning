library(xts)
library(zoo)
library (lubridate)

#Setting working directory 
path <- 'C:/Users/postgres/Downloads'
setwd(path)

# Load the CSV data into a data frame
j17 <- read.csv('J-17.csv ')

# Extract the relevant time series data
DailyHighDate <- j17$DailyHighDate

# Check for missing values
missing_values <- sum(is.na(DailyHighDate))

# Print the number of missing values
cat("Number of missing values in the time series data:", missing_values, "\n")

#drop the first column (sitename column) 
j17 <-j17[,-1]

# Convert date column to Date format
j17$DailyHighDate <- as.Date(j17$DailyHighDate)

# Remove duplicate rows
j17 <- j17[!duplicated(j17), ]

#Extract beginning and end dates
bgn <- as.Date(j17$DailyHighDate[1])
end <- as.Date(j17$DailyHighDate[length(j17$DailyHighDate)])

# Print the structure of the dataframe
str(j17)

#Print summary of the dataframe
summary(j17)

# Identify and replace outliers with NA
outliers <- which(j17$WaterLevelElevation > 703.2 | j17$WaterLevelElevation < 612.5)
j17$WaterLevelElevation[outliers] <- NA

#Check to see if there are missing values
any(is.na(j17)) #one outlier was found

# Remove rows with outliers
j17 <- j17 %>% drop_na()

##To visually inspect the time series to confirm the presence of outliers
#Extract the relevant time series data
time_series_j17 <- j17$WaterLevelElevation

# Plot the time series data
plot(time_series_j17, type = "l", xlab = "DailyHighDate", ylab = "WaterLevelElevation")

# Add points for outliers
outlier_indices <- c(5, 10, 15) # Replace with the actual outlier indices
points(outlier_indices, time_series_j17[outlier_indices], col = "red", pch = 19)

##########Autocorrelation Test########
# Perform autocorrelation test
# Select 10-year period for Autocorrelation Test
start_year <- 1933
end_year <- 2023
j17_90_years <- subset(j17, DailyHighDate >= start_year & DailyHighDate <= end_year)

# Extract the relevant time series data
time_series_90 <- j17_90_years$DailyHighDate

# Perform autocorrelation test
acf_data <- acf(time_series_90, lag.max = 10, plot = TRUE) #J-17 time series values at lag.max = 10, are positively correlated with the values at the current time point. 
                                                        
#Create a zoo object
j17_zoo <- zoo(j17$WaterLevelElevation, j17$DailyHighDate)

# Print the structure of the zoo object
str(j17_zoo)

# Plot the time series data using base R graphics
plot(j17_zoo, xlab = 'DailyHighDate', ylab = 'Water Level', main = "J17 Annual Time Series")

# Print the summary of the zoo object
summary(j17_zoo)

# Interpolate the missing values using na.approx
j17_zoo_interp <- na.approx(j17_zoo)

# Print the interpolated zoo object
print(j17_zoo_interp)

# Plot j17_zoo_interp time series data using base R graphics
plot(j17_zoo_interp, xlab = 'Year', ylab = 'Water Level', main = "J17 Time Series")


########Test for Stationarity
library(tseries)
adf.test(j17_zoo_interp) #if p-value < 0.05, then null hypothesis can be rejected, data is stationary.Meaning the data has constant mean, varaiance and autocorrrelation

########Decompose the Timeseries######

# Convert the dataframe to a time series object
ts_j17 <- ts(j17$WaterLevelElevation, start=c(2023-03-02, 1), end=c(1932-11-12, 32181), frequency=12)

# Calculate the 10-day moving average
ma_10d <- rollapply(ts_j17, width = 10, FUN = mean, align = "right")

# Print the result
print(ma_10d)

# Perform seasonal decomposition
j17_decomp <- decompose(ma_10d)

# Extract the components
trend <- j17_decomp$trend
seasonal <- j17_decomp$seasonal
random <- j17_decomp$random

# Visualize the components
library(ggplot2)

ggplot() +
  geom_line(aes(x=index(ts_j17), y=ts_j17), color='black', size=1) +
  geom_line(aes(x=index(trend), y=trend), color='red', size=1) +
  geom_line(aes(x=index(seasonal), y=seasonal), color='blue', size=1) +
  geom_line(aes(x=index(random), y=random), color='green', size=1) +
  labs(title="Seasonal Decomposition of Time Series", x="Year", y="Value") +
  theme_minimal()

# Plot the original time series data and the trend, seasonal, and residual components of the decomposition
plot(ts_j17, main="Original J17 Data")
plot(trend, main="Trend Component")
plot(seasonal, main="Seasonal Component")
plot(random, main="Residual Component")

########Statistical Test Analysis######
# Perform ADF test
library(urca)

result <- ur.df(ts_df, type="trend", lags=12)

# Print the test results
summary(result)




