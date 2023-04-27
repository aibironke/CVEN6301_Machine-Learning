library (lubridate)
library (forecast)
library (zoo)
library (xts)
library (imputeTS)
library (tseries)

#Setting working directory 
path <- 'D:/CVEN6301_Machine_Learning/Timeseries'
setwd(path)

# Load the CSV data into a data frame
j17 <- read.csv('J-17.csv ')
comal <- read.csv('Comal.csv ',skip = 28, header = TRUE)
sm <- read.csv('SanMarcos.csv ',skip = 28, header = TRUE)

#Trim data set
j17 <-j17[,-1]
comal<- comal[-1, -c(1, 2, 5)]
sm <- sm[-1, -c(1, 2, 5)]
X<-nrow(j17)
y<-nrow(comal)
Z<-nrow(sm)

# Convert date column to Date format
j17$DailyHighDate <- as.Date(j17$DailyHighDate)
comal$X.1 <- as.Date(comal$X.1,format = "%m/%d/%Y" )
sm$datetime <- as.Date(sm$datetime,format = "%m/%d/%Y" )

# Extract the relevant time series data
DailyHighDate <- j17$DailyHighDate
Datetime1 <- comal$X.1
Datetime2 <- sm$datetime

# Check for missing values
missing_values1 <- sum(is.na(DailyHighDate))
missing_values2 <- sum(is.na(Datetime1))
missing_values3 <- sum(is.na(Datetime2))

# Print the number of missing values
cat("Number of missing values in the time series data:", missing_values1, "\n")
cat("Number of missing values in the time series data:", missing_values2, "\n")
cat("Number of missing values in the time series data:", missing_values3, "\n")

#Extract begin and end for j-17 dataset
bgn1 <- as.Date(j17$DailyHighDate[length(j17$DailyHighDate)],format='%Y-%m-%d')
end1 <- as.Date(j17$DailyHighDate[1],format='%Y-%m-%d')
date1 <- seq.Date(bgn1,end1,'day')
pdate1 <- as.Date(j17$DailyHighDate,format='%Y-%m-%d')

#Extract begin and end for comal
bgn2 <- as.Date(comal$X.1[1])
end2 <- as.Date(comal$X.1[length(comal$X.1)])
date2 <- seq.Date(bgn2,end2,'day')
pdate2 <- as.Date(comal$datetime,format='%m/%d/%Y')

#Extract begin and end for sm
bgn3 <- as.Date(sm$datetime[1])
end3 <- as.Date(sm$datetime[length(sm$datetime)])
date3 <- seq.Date(bgn3,end3,'day')
pdate3 <- as.Date(sm$datetime,format='%m/%d/%Y')

#check structure of the data
str(j17)
str(comal)
str(sm)

# Convert the character column to numeric
comal$X.2 <- as.numeric(comal$X.2)
sm$X136526_00060_00003 <- as.numeric(sm$X136526_00060_00003)

#Reprint summary
summary(j17)
summary(comal)
summary(sm)

#Create a zoo object
j17_zoo <- zoo(j17$WaterLevelElevation, j17$DailyHighDate)
comal_zoo <- zoo(comal$X.2, comal$X.1)
sm_zoo <- zoo(sm$X136526_00060_00003, sm$datetime)

# Interpolate missing values, if any, using na.approx
j17_zoo_interp <- na.approx(j17_zoo)
comal_zoo_interp <- na.approx(comal_zoo)
sm_zoo_interp <- na.approx(sm_zoo)

# Convert the dataframe to a time series object
ts_j17 <- as.ts(j17_zoo_interp)
ts_comal <- as.ts(comal_zoo_interp)
ts_sm <- as.ts(sm_zoo_interp)

WL.tsj17 <- na_kalman(ts_j17,model= "StructTS")
WL.tscomal <- na_kalman(ts_comal,model= "StructTS") # to perform imputation
WL.sm <- na_kalman(ts_sm, model = "StructTS") # to perform imputation

#convert back to zoo
zoo_j17 <- zoo(WL.tsj17,date1)
zoo_comal <- zoo(WL.tscomal,date2)
zoo_sm <- zoo(WL.sm, date3)

#Plot the zoo objects
plot(zoo_j17, main='j17', xlab='Year',ylab='WaterLevel(ft)')
plot(zoo_comal, main='comal', xlab='Year',ylab='WaterLevel(ft)') 
plot(zoo_sm, main='San_Marcos', xlab='Year',ylab='WaterLevel(ft)')

#cross correlattion function
ccf(zoo_j17, zoo_comal)
ccf(zoo_j17, zoo_sm)
ccf(zoo_comal, zoo_sm)

# Print the start and end values of the values column
cat("Start value: ", coredata(sm_zoo)[1], "\n")
cat("End value: ", coredata(sm_zoo)[length(coredata(sm_zoo))], "\n")

# Check for missing values in the zoo object
missing_values <- is.na(sm_zoo)

# Print the positions of the missing values
print(which(missing_values))

#Autocorrelation Test
acf(zoo_sm)

########Test for Stationarity
#if p-value < 0.05, then null hypothesis can be rejected, data is stationary.
# Meaning the data has constant mean, varaiance and autocorrrelation.

library(tseries)
adf.test(sm_zoo)

#	Augmented Dickey-Fuller Test
#data:  sm_zoo
#Dickey-Fuller = -7.2784, Lag order = 29, p-value = 0.01
#alternative hypothesis: stationary


########Decompose the Timeseries######

# Convert the zoo object to a data frame
library(ggplot2)
sm_ts <- fortify.zoo(sm_zoo)

# Print the resulting data frame
print(sm_ts)

# Convert the dataframe to a time series object
ts_sm <- ts(sm_ts, start=c(2023-04-09, 1), end=c(1956-05-26, 24425), frequency=12)

# Calculate the 10-day moving average
ma_10d <- rollapply(ts_sm, width = 10, FUN = mean, align = "right")

# Print the result
print(ma_10d)

# Perform seasonal decomposition
sm_decomp <- decompose(ma_10d)

# Extract the components
trend <- sm_decomp$trend
seasonal <- sm_decomp$seasonal
random <- sm_decomp$random

# Plot the trend, seasonal, and residual components of the decomposition
plot(trend, main="Trend Component")
plot(seasonal, main="Seasonal Component")
plot(random, main="Residual Component")


#####Cross Correlation Test
library (stats)

#import data sets
x<-comal_zoo_interp


# Plot the two time series data
ggplot() + 
  geom_line(aes(seq(0, 20, by=0.1), x), col = 'blue', size = 1) + 
  geom_line(aes(seq(0, 20, by=0.1), y), col = 'red', size = 1)
