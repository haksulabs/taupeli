# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:21:33 2023

taupeli logit with pymer

@author: t
"""
import os
os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lm
from pymer4.models import Lmer
from sklearn.preprocessing import StandardScaler






df = pd.read_csv('taupelidata_corners_pilotit.csv')

scaler = StandardScaler()

df['abs_ttcdiff'] = np.abs(df.ttcdiff)
df['abs_ttcdiff_norm'] = scaler.fit_transform(df[['abs_ttcdiff']])

df['minttc_norm'] = scaler.fit_transform(df[['minttc']])

df['abs_v0'] = np.abs(df.v0)
df['abs_v1'] = np.abs(df.v1)
df['max_v'] = df[['abs_v0','abs_v1']].max(axis=1)
df['min_v'] = df[['abs_v0','abs_v1']].min(axis=1)

df['min_v_norm'] = scaler.fit_transform(df[['min_v']])
df['max_v_norm'] = scaler.fit_transform(df[['max_v']])


df['x0_end'] = np.sign(df.x0) * (np.abs(df.x0) - df.v0*0.5)
df['x1_end'] = np.sign(df.x1) * (np.abs(df.x1) - df.v1*0.5)

df['y0_end'] = np.sign(df.y0) * (np.abs(df.y0) - df.v0*0.5)
df['y1_end'] = np.sign(df.y1) * (np.abs(df.y1) - df.v1*0.5)

df['xseparation'] = np.abs(df.x1_end - df.x0_end)
df['yseparation'] = np.abs(df.y1_end - df.y0_end)

df['xenddif'] = np.abs(np.abs(df.x1_end) - np.abs(df.x0_end))
df['yenddif'] = np.abs(np.abs(df.y1_end) - np.abs(df.y0_end))

orig_df = df.copy()



print('condition 23')
df =orig_df[orig_df['condition']==23]

model = Lm("correct ~ 1 +  abs_ttcdiff", data=df, family = 'binomial')
print(model.fit())

model = Lm("correct ~ 1 +  abs_ttcdiff + xenddif", data=df, family = 'binomial')
print(model.fit())



print('startpos 11')
df1 = df[df['startpos'].isin([11,55])]
model = Lm("correct ~ 1 +  abs_ttcdiff + xenddif", data=df1, family = 'binomial')
print(model.fit())




print('startpos 15')
df2 = df[df['startpos'].isin([15,51])]
model = Lm("correct ~ 1 +  abs_ttcdiff + xenddif", data=df2, family = 'binomial')
print(model.fit())

# # condition 12
# print('condition 23')
# df =orig_df[orig_df['condition']==23]



# model = Lm("correct ~ 1 +  abs_ttcdiff", data=df, family = 'binomial')
# print(model.fit())

# model = Lm("correct ~ 1 +  abs_ttcdiff + xenddif", data=df, family = 'binomial')
# print(model.fit())







# multi level models

# print('--------------------------------------------')
# print('multilevel model')

# model = Lmer("correct ~ abs_ttcdiff + (1|name) ", data=df, family = 'binomial')

# print(model.fit())

# print('--------------------------------------------')
# print('multilevel model')

# model = Lmer("correct ~ abs_ttcdiff + xdif + (1|name) ", data=df, family = 'binomial')

# print(model.fit())


# print('--------------------------------------------')
# print('multilevel model')

# model = Lmer("correct ~ abs_ttcdiff + ydif + (1|name) ", data=df, family = 'binomial')

# print(model.fit())





# # condition 23 ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤


# print('condition 23')

# df =orig_df[orig_df['condition']==23]

# model = Lm("correct ~ 1 +  abs_ttcdiff + xdif + ydif", data=df, family = 'binomial')
# print(model.fit())


# # multi level models
# print('--------------------------------------------')
# print('multilevel model')

# #model = Lmer("correct ~ abs_ttcdiff + xdif + ydif + (abs_ttcdiff|name) + (xdif|name) + (ydif|name) ", data=df, family = 'binomial')

# print(model.fit())




