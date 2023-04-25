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

#remove pracitce trials
df = df[df['n_trials'] >20]


df['abs_ttcdiff'] = np.abs(df.ttcdiff)
df['abs_ttcdiff_norm'] = scaler.fit_transform(df[['abs_ttcdiff']])

df['minttc_norm'] = scaler.fit_transform(df[['minttc']])

df['abs_v0'] = np.abs(df.v0)
df['abs_v1'] = np.abs(df.v1)
df['max_v'] = df[['abs_v0','abs_v1']].max(axis=1)
df['min_v'] = df[['abs_v0','abs_v1']].min(axis=1)
df['delta_v'] = df.max_v-df.min_v
df['delta_v_norm'] = scaler.fit_transform(df[['delta_v']])

df['min_v_norm'] = scaler.fit_transform(df[['min_v']])
df['max_v_norm'] = scaler.fit_transform(df[['max_v']])

df['v_div_v'] = df['max_v']/df['min_v']
df['sum_v'] = df['max_v'] + df['min_v']


df['x0_end'] = np.sign(df.x0) * (np.abs(df.x0) - df.v0*0.5)
df['x1_end'] = np.sign(df.x1) * (np.abs(df.x1) - df.v1*0.5)
df['abs_x0'] = np.abs(df.x0) - df.v0*0.5
df['abs_x1'] = np.abs(df.x1) - df.v0*0.5
df['max_x'] = df[['abs_x0','abs_x1']].max(axis=1)
df['min_x'] = df[['abs_x0','abs_x1']].min(axis=1)
df['x_div_x'] = df['max_x']/df['min_x'] 


df['y0_end'] = np.sign(df.y0) * (np.abs(df.y0) - df.v0*0.5)
df['y1_end'] = np.sign(df.y1) * (np.abs(df.y1) - df.v1*0.5)

df['xseparation'] = np.abs(df.x1_end - df.x0_end)
df['yseparation'] = np.abs(df.y1_end - df.y0_end)
df['xseparation_norm'] = scaler.fit_transform(df[['xseparation']])
df['yseparation_norm'] = scaler.fit_transform(df[['yseparation']])

df['tot_separation'] = np.sqrt(df['xseparation']**2 + df['yseparation']**2)
df['tot_separation_norm'] = scaler.fit_transform(df[['tot_separation']])

df['xenddif'] = np.abs(np.abs(df.x1_end) - np.abs(df.x0_end))
df['yenddif'] = np.abs(np.abs(df.y1_end) - np.abs(df.y0_end))

df['xenddif_norm'] = scaler.fit_transform(df[['xenddif']])
df['yenddif_norm'] = scaler.fit_transform(df[['yenddif']])

df['delta_x_norm'] = df['xenddif_norm']

df['totdif'] = np.sqrt(df['xenddif']**2 + df['yenddif']**2)
df['totdif_norm'] = scaler.fit_transform(df[['totdif']])

df['stagger'] = df['startpos'].isin([15,51])
df['stagger'] = df['stagger'].astype(int)
orig_df = df.copy()

# con=23
# print('condition ', con)
# df =orig_df[orig_df['condition']==con]

# model = Lm("correct ~ 1 +  abs_ttcdiff_norm", data=df, family = 'binomial')
# print(model.fit())

# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df, family = 'binomial')
# print(model.fit())



# print('startpos 11')
# df1 = df[df['startpos'].isin([11,55])]
# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df1, family = 'binomial')
# print(model.fit())




# print('startpos 15')
# df2 = df[df['startpos'].isin([15,51])]
# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df2, family = 'binomial')
# print(model.fit())

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



con = [12]
stag = 1
df =orig_df[orig_df['condition'].isin(con)]
df =df[df['stagger']==stag]            
dfcon = df.copy()

print('--------------------------------------------')
print('multilevel models')
print('condition:', con)

#print('basic model with only ttcdiff')
#model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
#print(model.fit())

#print('model ttcdiff  + yseparation')

model = Lmer("correct ~ abs_ttcdiff_norm + tot_separation_norm +  delta_x_norm + (1|name) ", data=df, family = 'binomial')
#model = Lmer("correct ~  tot_separation_norm +  delta_x_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
#model = Lm("correct ~ abs_ttcdiff_norm+ tot_separation_norm +  totdif_norm +  delta_v_norm ", data=df, family = 'binomial')
print(model.fit())

#model2 = Lmer("correct ~ abs_ttcdiff_norm + sum_v +    (1|name) ", data=df, family = 'binomial')
#print(model2.fit())

# res_df = pd.DataFrame()
# pred_candidates = ['delta_v_norm','delta_x_norm','x_div_x','v_div_v','sum_v','tot_separation_norm']
# for pred in pred_candidates:
#     df['predictor'] = df[pred]
#     model = Lmer("correct ~ abs_ttcdiff_norm + predictor + (1|name) ", data=df, family = 'binomial')
#     model.fit()
#     print(pred + ':' + str(model.AIC))
#     d = {'predictor': pred, 'AIC': model.AIC}
#     d_df = pd.DataFrame([d])
#     res_df =  pd.concat([res_df, d_df], ignore_index=True)
    
    
# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')


############

# print('startpos 11 nostagger')
# df = dfcon[dfcon['startpos'].isin([11,55])]

# print('basic model with only ttcdiff')
# model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')




# ###########
# print('startpos 15 stagger')


# df = dfcon[dfcon['startpos'].isin([15,51])]

# print('basic model with only ttcdiff')
# model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())






# con = 12
# df =orig_df[orig_df['condition']==con]
# dfcon = df.copy()

# print('------con 12--------------------------------------')
# print('nostagger')
# df = dfcon[dfcon['startpos'].isin([11,55])]

# model = Lmer("correct ~ abs_ttcdiff_norm + xseparation_norm + xenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')

# print('stagger')

# df = dfcon[dfcon['startpos'].isin([15,51])]

# model = Lmer("correct ~ abs_ttcdiff_norm + xseparation_norm + xenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())







