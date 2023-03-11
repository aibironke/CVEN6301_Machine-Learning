# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:56:43 2023

@author:Ademola Ibironke
"""

#load libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#Setting working directory
path = 'D:/CVEN6301_Machine_Learning/Project_2'
os.chdir(path)

#Reading the dataset
a = pd.read_csv('Wells_Final.csv')
a
# Copying data in a new dataframe object
b = a.copy()

# Arranging variables to perform regression
Y = a['Avg_Fl'] # strength is the Y variable

#Extracting columns to be used as X variables
X = a.iloc[:, 5:48]

#selecting parameters based on hit and trial
cols = [8,11,14,17,20,23,26,29,32,35,38,40,43,45,46,47]
X = a[a.columns[cols]]

#normalizing the parameters
X = (X - X.min())/(X.max()-X.min())

########################CORRELATION###########################
# Compute correlations and plot correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
 
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#selecting parameters based correlation shown on heatmap
cols = [8,11,17,20,26,29,38,40,43,45]
X = a[a.columns[cols]]

#normalizing the parameters
X = (X - X.min())/(X.max()-X.min())

# Compute correlations and plot correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
 
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
##############################################################


#Setting Flouride Threshold as Indicator of exceedence or otherwise
threshold = 1.2
b['Fl_Ind'] = np.where(a['Avg_Fl'] >= threshold, 1, 0)
Y = b['Fl_Ind']

 
#Splitting the data to training anad testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=99)
 
# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means low regularization

#Training the data
# fit the model with data
logreg.fit(X_train,y_train)

# Make Predictions
y_pred=logreg.predict(X_train) # Make Predictions
yprob = logreg.predict_proba(X_train) #test output probabilities


# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_
 
# Create a confusion Matrix and plot it
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

# Evaluate the model
acc = accuracy_score(y_train, y_pred) #to measure the overall correctness of the model
precision = precision_score(y_train, y_pred) #measure the fraction of true positive predictions among all the positive predictions made by the mode
recall = recall_score(y_train, y_pred) #measures the fraction of true positive predictions among all the actual positive instances in the dataset
f1 = f1_score(y_train, y_pred) #to evaluate the performance of the model
 
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")

 
# ROC Curve
y_pred_proba = logreg.predict_proba(X_train)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
auc = metrics.roc_auc_score(y_train, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

 

#Testing the data

# Make Predictions
y_pred=logreg.predict(X_test) # Make Predictions

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

# Create a confusion Matrix and plot it
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

 
# Evaluate the model
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

 
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")


# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

###
logreg.fit(X_train,y_train)
pred_LR = logreg.predict(X) # Make Predictions
proba_LR = logreg.predict_proba(X)[::,1]

 
d = {'WellID':a.StateWellNumber, 'LatDD':a.LatDD, 'LonDD':a.LonDD, 'predLR':proba_LR}
final = pd.DataFrame(d)
final.to_csv('LRPreds.csv',index = False)