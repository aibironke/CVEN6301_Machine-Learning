# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:40:03 2023

@author: Ademola Ibironke
"""

# load libraries
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#Setting working directory
path = 'D:\CVEN6301_Machine_Learning\Project_1'
os.chdir(path)

#Reading the dataset
a = pd.read_csv('Wells_Params.csv')
a.head(5) # explore first 5 rows
a.columns #view all columns header

# Compute correlations matrix of all varaiables
corr = a.corr(method='kendall')
corr #to view correlation matrix in Pyhton Console
pd.DataFrame.to_csv(corr,'correl.csv') # Write correlation matrix to csv file
# source:https://seaborn.pydata.org/examples/many_pairwise_correlations.html

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#CORRELATION ANALYSIS
# Compute Mutual Information Criteria for Y and X variables
Y = a['Avg_Nit'] # Avg_Nitrate is the Y variable
#Extracting columns to be used as X variables
cols = [7,8,9,17,18]
X = a[a.columns[cols]]

# Compute correlations and plot correlation matrix
corr2 = X.corr() #We want our 2.5% and 97.5% values be either be all negative or all positive

# Generate a mask for the upper triangle
mask2 = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap for independent variables with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#MULTICOLLINEARITY
# Compute Variance Inflation Factors for X
#VIF is measure of the amount of multicollinearity
#A factor with a VIF higher than 10 indicates a problem of multicollinearity existed (Dou et al.2019).
#https://www.tandfonline.com/doi/full/10.1080/17538947.2020.1718785
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif #view VIF



#Splitting the data to training anad testing
#splitting 75% for training  & 25% for testing 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=12)


#Performing random forest regressor for feature importance
rf = RandomForestRegressor(n_estimators=100) #algorithm to predict continous target variable to improve accuracy and robutness of model.
rf.fit(X_train, y_train)
rf.feature_importances_
plt.barh(X.columns, rf.feature_importances_)

#Training data
#Linear Regression Model
X_train = sm.add_constant(X_train) #Adding a constant
model = sm.OLS(y_train,X_train)
res = model.fit()
print(res.summary())


#Predict using the same model
pred = res.fittedvalues.copy()
err = y_train - pred

#Create dataframe for observed and predicted y variables based y_train
zz = pd.DataFrame(list(zip(y_train, pred)),columns=['y_obs','y_pred'])
zz['Obs_Ind'] = np.where(zz['y_obs'] > 3, 1, 0)
zz['Pred_Ind'] = np.where(zz['y_pred'] > 3, 1, 0)
zz

#Create a contingency Table for observed and predicted y based on y_train
pd.crosstab(index=zz['Obs_Ind'], columns=zz['Pred_Ind'])


#Testing data
X_test = sm.add_constant(X_test) #Adding a constant
yt_pred = res.predict(X_test)

#Create dataframe for observed and predicted y variables based on y_test
zz2 = pd.DataFrame(list(zip(y_test, yt_pred)),columns=['yt_obs','yt_pred'])
zz2['Obs_Ind'] = np.where(zz2['yt_obs'] > 3, 1, 0)
zz2['Pred_Ind'] = np.where(zz2['yt_pred'] > 3, 1, 0)
zz2

#Create a contingency Table for observed and predicted y based on y_test
pd.crosstab(index=zz2['Obs_Ind'], columns=zz2['Pred_Ind'])

# Obs-Predicted plot
plt.plot(y_test,yt_pred,'bo')
plt.plot(y_test, y_test + 0, linestyle='solid')
plt.xlabel('Obs_Ind' )
plt.ylabel('Pred_Ind' )
plt.grid()


#Breusch Pagan Test for homoskedasticity
BP = sm.stats.diagnostic.het_breuschpagan(err,X_train)

# tests of Normality
sm.qqplot(err,line='s')
plt.grid()

sm.stats.diagnostic.kstest_normal(err, dist='norm')

# Autocorrelation testing
sm.stats.diagnostic.acorr_breusch_godfrey(res) #need to provide regression model

#from sklearn.linear_model import LinearRegression
#mod = LinearRegression()
#mod.fit(X_train,y_train) #Fitting the mode using training data
#Prediction for testing data and training data
#y_test_pred = mod.predict(X_test)
#y_train_pred = mod.predict(X_train)