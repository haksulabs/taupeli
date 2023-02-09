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
import statsmodels.api as sm




df = pd.read_csv('taupelidata_corners_pilotit.csv')

#name = 'Samuel'
#name = 'kh37_corners'
name = 'tero_eta'


df1 = df[df['name']==name]   

#df1= df.copy()


X = pd.DataFrame()
X['abs_ttcdiff'] = np.abs(df1.ttcdiff)
X['max_speed'] = np.abs(df1[["v0", "v1"]]).max(axis=1)
X['min_speed'] = np.abs(df1[["v0", "v1"]]).min(axis=1)
X['reply_time'] = df1['reply_time']
X['minttc'] = df1['minttc']
X['xdiff'] = np.abs(df1.x0 + df1.v0*0.5) + np.abs(df1.x1 - df1.v1*0.5)



y = df1.correct

# set baseline probability to 0.5, corresponds to intercept = 0 
X.insert(0,'intercept',0)

# select features to use in the model
#features = ['intercept','abs_ttcdiff','max_speed','min_speed']
features = ['intercept','abs_ttcdiff','xdiff']

X_train,X_test, y_train, y_test = train_test_split(X[features], y, train_size = 0.9)


model = LogisticRegression(fit_intercept=False)
model.fit(X_train,y_train)

fitted_ytest = expit(X_test.abs_ttcdiff * model.coef_[0,1] + model.intercept_)

y_p = model.predict(X_test)
y_prob = model.predict_proba(X_test)


d = {'X_test':X_test.abs_ttcdiff,'y_test': y_test, 'y_predicted': y_p,'y_probability':y_prob[:,1]  }
df_modelpred = pd.DataFrame(d)

x = np.linspace(0, 1, 100)
# sanity check, calc logit with abs_ttc_diff 
fitted_1 = expit(x * model.coef_[0,1] + model.intercept_)


plt.scatter(X_test.abs_ttcdiff, y_test, label="data", color="red")
plt.plot(x, fitted_1, '-', label="Logistic Regression Model with only abs_ttcdiff", color="red")

plt.scatter(X_test.abs_ttcdiff, y_prob[:,1], label="Logistic Regression predicted probability", color="blue")


plt.legend()
plt.show()


# use statsmodel instaed to get p-values etc

features = ['abs_ttcdiff', 'xdiff']
Xs = X[features]

smodel = sm.Logit(y, Xs)
result = smodel.fit(method='newton')
print(result.summary())
