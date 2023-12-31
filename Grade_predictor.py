# -*- coding: utf-8 -*-
"""Untitled34.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bUf-_QXBW7cHxg1wNHnu0zC1msqPtyeN

Importing the Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""

data=pd.read_csv('Grade_Predictor_Data.csv')
X=data.iloc[:,1:4].values

int_features=[65,70,11,4]
y=int_features.pop()
features=np.array(int_features)

print(features.reshape(1,-1))

Y=data.iloc[:,y].values
print(Y)

"""Splitting into Training and Test set"""

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(X_train)

"""Decision Tree Regressor"""

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X_train,Y_train)

#print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))

"""New Predictions"""

regressor.predict(sc.transform(features.reshape(1,-1)))

"""Score test"""

#from sklearn.metrics import r2_score
#print(r2_score(Y_test,y_pred))