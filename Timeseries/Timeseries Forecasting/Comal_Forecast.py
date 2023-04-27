# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:17:19 2023

@author: postgres
"""

#libraries
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

#Setting working directory
path = 'D:/CVEN6301_Machine_Learning/Timeseries'
os.chdir(path)

# Load the data
comal = pd.read_csv('Comal.csv')

# read the CSV file, skipping the first 27 rows and columns 1, 2, and 5
comal = pd.read_csv('Comal.csv', skiprows=range(28), usecols=lambda x: x not in [1, 2, 5])

# read the 28th row as the header
comal = comal.rename(columns=comal.iloc[0]).drop(comal.index[0])
# drop columns 1, 2 & 5
comal = comal.drop(comal.columns[[0, 1, 4]], axis=1)

# Clean the data
comal.dropna(inplace=True)

#Transform data: convert date string to datetime format
comal["datetime"] = pd.to_datetime(comal['datetime'])

# Set the date column as the index
comal.set_index('datetime', inplace=True)

# # Specify start and end datetime index
start_date = pd.to_datetime("2013-01-01 00:00:00")
end_date = pd.to_datetime("2023-01-01 00:00:00")

# Filter the dataframe to include only rows between start and end dates
comal = comal.loc[start_date:end_date]

# Split the data into training, validation, and test sets
train_data = comal.iloc[:int(len(comal)*0.4)] #selects the first 40% of the data (rows) for the training set
val_data = comal.iloc[int(len(comal)*0.4):int(len(comal)*0.7)] # selects the rows between the first 40% and 70% for validation
test_data = comal.iloc[int(len(comal)*0.7):]#selects the remaining 30% of the data for the test set


# Normalize the data by Scaling between 0 and 1
# Helps to reduce the impact of extreme values and make it easier to identify patterns and relationships
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Create input and output sequences for the LSTM model
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i]) #appends a sequence of seq_length values from the data array to the X array
        y.append(data[i]) #appends the next value in the data array to the y array
    X = np.array(X)
    y = np.array(y)
    return X, y

seq_length = 30 # Number of days to use as input to the model
X_train, y_train = create_sequences(train_scaled, seq_length)
X_val, y_val = create_sequences(val_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Define the LSTM model architecture
model = Sequential() #creates linear stack of layers, where you can add one layer at a time using the "add()" fnc
model.add(LSTM(units=64, input_shape=(seq_length, 1))) #LSTM layer added w/64 LSTM units and input shape "seq_lenth" and the 1 indicates that there is only one feature (i.e., water level elevation) in the input. 
model.add(Dense(1)) #Dense specifies the no of ouptput neurons

# Compile the model with "mean_squared_error" loss function and "adam" optimizer
#this helps to specify how the model's performance will be measured and how the weights will be updated during optimization.
#'mean_squared_error'  minimizes the difference between the predicted and actual values
# "adam" adjusts the weights of the model during training to minimize the loss function
model.compile(loss='mean_squared_error', optimizer='adam')

# Set up early stopping callback
#the model will be trained on the training set for a maximum of 100 epochs, but will stop early if the validation loss does not improve for 10 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)


# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test data
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the scaled data to get the actual values
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot the actual and predicted values for the test data
import matplotlib.pyplot as plt
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.show()