#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:04:36 2022

@author: t
"""


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#df = pd.read_csv('taupelidata_pilotit.csv')
#df = pd.read_csv('taupelidata_noise.csv')
#df = pd.read_csv('taupelidata_randomloc.csv')

df = pd.read_csv('taupelidata_corners_pilotit.csv')
df = df[df['n_trials']>20]
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


df['x0_end'] = np.sign(df.x0) * (np.abs(df.x0) - df.v0*0.5)
df['x1_end'] = np.sign(df.x1) * (np.abs(df.x1) - df.v1*0.5)

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

df['totdif'] = np.sqrt(df['xenddif']**2 + df['yenddif']**2)
df['totdif_norm'] = scaler.fit_transform(df[['totdif']])

df['stagger'] = df['startpos'].isin([15,51])
df['stagger'] = df['stagger'].astype(int)
orig_df = df.copy()
df['x0_end'] = np.sign(df.x0) * (np.abs(df.x0) - df.v0*0.5)
df['x1_end'] = np.sign(df.x1) * (np.abs(df.x1) - df.v1*0.5)
df['x0_kauempana'] = np.sign((np.abs(df['x0_end']) - np.abs(df['x1_end'])))                             
df['overtake'] = (df['x0_kauempana'] == np.sign(df['ttcdiff']))
df['overtake'] = df['overtake'].astype(int)






df['bins'] = pd.cut(df['abs_ttcdiff'], bins=10)
df['binmid'] = df['bins'].apply(lambda x: x.mid)
df['bins'].value_counts()


df['ec_bins'] = pd.cut(df['tot_separation'], bins=10)
df['ec_binmid'] = df['ec_bins'].apply(lambda x: x.mid)
df['ec_bins'].value_counts()



foo = df['bins'].unique()
foo = foo.sort_values()
res=pd.DataFrame()

for i,f in enumerate(foo):
    print(f)
    fmid = f.mid
 #   print(len(df[df['bins']==f]))
    ok = len(df[ (df['bins']==f) & (df['correct'] == 1)])
    nall = (len(df[df['bins']==f]))
    print('correct:' + str(ok/nall))
    
    d = {'ttcmid': fmid, 'accuracy': ok/nall}
    res = res.append(d, ignore_index=True)
    
    
foo = df['ec_bins'].unique()
foo = foo.sort_values()
ec_res=pd.DataFrame()

for i,f in enumerate(foo):
    print(f)
    fmid = f.mid
 #   print(len(df[df['bins']==f]))
    ok = len(df[ (df['ec_bins']==f) & (df['correct'] == 1)])
    nall = (len(df[df['ec_bins']==f]))
    print('correct:' + str(ok/nall))
    
    d = {'ttcmid': fmid, 'accuracy': ok/nall}
    ec_res = ec_res.append(d, ignore_index=True)
    
