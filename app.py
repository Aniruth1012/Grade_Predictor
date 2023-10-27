# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:00:12 2023

@author: admin
"""

import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('webindex.html')

data=pd.read_csv('Grade_Predictor_Data.csv')
X=data.iloc[:,1:4].values





@app.route('/predict',methods=['GET', 'POST'])
def predict():
    
    int_features=[float(x) for x in request.form.values()]
    y=int_features.pop()
    features=[np.array(int_features)]
    print(y)
    y=int(y)
    desired=65+y-5
    if(y==4):
        desired=83
    desired=chr(desired)
    print(desired)
    Y=data.iloc[:,y].values
    print(Y)
    
    """Splitting into Training and Test set"""
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

    """Feature Scaling"""

    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)

    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=500,random_state=0)
    model.fit(X_train,Y_train)

    features=sc.transform(features)
    print(features)
    prediction=model.predict(features)
    output=np.ceil(prediction)
    return render_template('webindex.html',prediction_text='Predicted mark for %s' %desired+' grade {}'.format(prediction))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)