# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 01:58:01 2023

Logistic regression model for taupeli 

first to do: 
1) 1kh 
2) predictor is ttc-diff



@author: t
"""



import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import expit




df = pd.read_csv('taupelidata_corners_pilotit.csv')
df1 = df[df['name']=='kh28_corners']   


X_train,X_test, y_train, y_test = train_test_split(np.abs(df1[['ttcdiff']]), df1.correct, train_size = 0.9)

# add constant column to represent intercept to 0 
X_train.insert(0,'intercept',0)
X_test.insert(0,'intercept',0)

model = LogisticRegression(fit_intercept=False)
model.fit(X_train,y_train)



fitted = expit(X_test.ttcdiff * model.coef_[0,1] + model.intercept_)
#plt.scatter(X_test['ttcdiff'],y_test,marker='+', color='red')
#plt.plot(X_test, fitted, label="Logistic Regression Model", color="red", linewidth=3)


y_p = model.predict(X_test)
y_prob = model.predict_proba(X_test)


d = {'X_test':X_test.ttcdiff,'y_test': y_test, 'y_predicted': y_p,'y_probability':y_prob[:,1]  }
df_test = pd.DataFrame(d)

x = np.linspace(0, 1, 100)
fitted = expit(x * model.coef_[0,1] + model.intercept_)

plt.scatter(X_test.ttcdiff, y_test, label="data", color="red", zorder=20)
plt.plot(x, fitted, '-', label="Logistic Regression Model", color="red")
plt.legend()
plt.show()
