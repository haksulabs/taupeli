# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 04:11:51 2023
mixed effects model testings

@author: t
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset('dietox','geepack').data

#model = smf.mixedlm('Weight ~ Time', data, groups = data['Pig'])
#res = model.fit(method=['lbfgs'])

model2 = smf.mixedlm('Weight ~ Time', data, groups = data['Pig'], re_formula="~Time")
res2 = model2.fit(method=['lbfgs'])
print(res2.summary())
