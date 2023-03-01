# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 01:39:21 2023

@author: postgres
"""




# Load Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#Setting working directory
path = 'D:\CVEN6301_Machine_Learning\Project_1'
os.chdir(path)

#Reading the dataset
a = pd.read_csv('Wells_Params.csv')
a.head(5) # explore first 5 rows

# Arranging variables to perform regression
#Extracting columns to be used as X variables
cols = [7,8,9,17,18]
X = a[a.columns[cols]]
Y = a['Avg_Nit'] # strength is the Y variable

# Scale the independent variables using StandardScaler()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Replace the original independent variables with the scaled variables in the DataFrame
X.iloc[:, :5] = X_scaled

# Define the threshold for reclassifying y
threshold = 3

# Reclassify y based on the threshold
y_reclassified = np.where(Y >= threshold, 1, 0)

# Fit a logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_scaled, y_reclassified)

# Make predictions on the original y values
y_pred = lr_model.predict_proba(X_scaled)[:, 1]
y_pred_reclassified = np.where(y_pred >= threshold, 1, 0)

# Evaluate the model
acc = accuracy_score(y_reclassified, y_pred_reclassified)
precision = precision_score(y_reclassified, y_pred_reclassified)
recall = recall_score(y_reclassified, y_pred_reclassified)
f1 = f1_score(y_reclassified, y_pred_reclassified)

print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")



# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_reclassified,test_size=0.30,random_state=10)

# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means no regularization

# fit the model with data
logreg.fit(X_train,y_train)


# Make Predictions
y_pred=logreg.predict(X_test) # Make Predictions
yprob = logreg.predict_proba(X_test) #test output probabilities
zz = pd.DataFrame(yprob)
zz.to_csv('nit_a.csv')

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

# Write the data to a file
keys = list(X.columns)
keys.append('Intercept')

vals = logreg.coef_.tolist()
vals = list(chain.from_iterable(vals))
intcept = float(logreg.intercept_)
vals.append(intcept)

par_dict = dict(zip(keys,vals))
with open('pars.txt','w') as data: 
      data.write(str(par_dict))

# Create a confusion Matrix and plot it
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

# Evaluate the model
acc = accuracy_score(y_reclassified, y_pred_reclassified)
precision = precision_score(y_reclassified, y_pred_reclassified)
recall = recall_score(y_reclassified, y_pred_reclassified)
f1 = f1_score(y_reclassified, y_pred_reclassified)

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