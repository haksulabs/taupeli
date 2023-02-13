# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:46:20 2023

check logistical regression significances using statsmodel 


@author: t
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('taupelidata_corners_pilotit.csv')

#name = 'Samuel'
#name = 'kh37_corners'
#name = 'tero_eta'

#data = df[df['name']==name].copy()
data = df.copy()

#data = sm.add_constant(df)
scaler = StandardScaler()

data['abs_ttcdiff'] = np.abs(data.ttcdiff)
data['abs_ttcdiff_norm'] = scaler.fit_transform(data[['abs_ttcdiff']])

data['minttc_norm'] = scaler.fit_transform(data[['minttc']])

data['abs_v0'] = np.abs(data.v0)
data['abs_v1'] = np.abs(data.v1)
data['max_v'] = data[['abs_v0','abs_v1']].max(axis=1)
data['min_v'] = data[['abs_v0','abs_v1']].min(axis=1)

data['min_v_norm'] = scaler.fit_transform(data[['min_v']])
data['max_v_norm'] = scaler.fit_transform(data[['max_v']])

#data['xdiff'] = np.abs(data.x0 + data.v0*0.5) + np.abs(data.x1 - data.v1*0.5)
#data['xdiff_norm'] = scaler.fit_transform(data[['xdiff']])

data['ydiff'] = (data.y0 - data.y1)
data['ydiff_norm'] = scaler.fit_transform(data[['ydiff']])

data['xdiff'] = (data.x0 - data.x1)
data['xdiff_norm'] = scaler.fit_transform(data[['xdiff']])

logit_model_basic = sm.Logit(data['correct'], data[['xdiff_norm']])
logit_model_basic2 = sm.Logit(data['correct'], data[['abs_ttcdiff_norm']])

logit_model = sm.Logit(data['correct'], data[['abs_ttcdiff_norm', 'minttc_norm','max_v_norm','min_v_norm','xdiff_norm','ydiff_norm']])



result_basic = logit_model_basic.fit()
result_basic2 = logit_model_basic2.fit()

result = logit_model.fit()



print(result_basic.summary())
print(result_basic2.summary())

print(result.summary())
