# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:21:38 2023

Here we precalculate everything that can be precalculated.

@author: t
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit

def corners_preprocess(df_orig):
#preprocess and precalculate stuff
    scaler = StandardScaler()
    df = df_orig[df_orig['n_trials'] >20].copy()
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
    df['v_div_v'] = df['min_v']/df['max_v']
    df['v_div_v_norm'] = scaler.fit_transform(df[['v_div_v']])
    df['sum_v'] = df['max_v'] + df['min_v']
    df['sum_v_norm'] = scaler.fit_transform(df[['sum_v']])
    df['x0_end'] = np.sign(df.x0) * (np.abs(df.x0) - df.abs_v0*0.5)
    df['x1_end'] = np.sign(df.x1) * (np.abs(df.x1) - df.abs_v1*0.5)
    df['abs_x0'] = np.abs(df.x0) - df.abs_v0*0.5
    df['abs_x1'] = np.abs(df.x1) - df.abs_v1*0.5
    df['max_x'] = df[['abs_x0','abs_x1']].max(axis=1)
    df['min_x'] = df[['abs_x0','abs_x1']].min(axis=1)
    df['delta_x_end'] = df['max_x'] - df['min_x'] 
    df['delta_x_end_norm'] =scaler.fit_transform(df[['delta_x_end']])    
    df['x_div_x'] = df['min_x']/ df['max_x']
    df['x_div_x_norm'] = scaler.fit_transform(df[['x_div_x']])    
    df['y0_end'] = np.sign(df.y0) * (np.abs(df.y0) - df.abs_v0*0.5)
    df['y1_end'] = np.sign(df.y1) * (np.abs(df.y1) - df.abs_v1*0.5)
    df['abs_y0'] = np.abs(df.y0) - df.abs_v0*0.5
    df['abs_y1'] = np.abs(df.y1) - df.abs_v1*0.5
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
    df['sigma'] = df['x0_end']**2 + df['x1_end']**2
    df['ttc_0'] =np.abs(df['x0']/df['v0'])
    df['ttc_1'] =np.abs(df['x1']/df['v1'])
    df['ttc_f'] = (np.sign(df['abs_v0']-df['abs_v1'])+1)*0.5*df['ttc_0'] + (np.sign(df['abs_v1']-df['abs_v0'])+1)*0.5*df['ttc_1']
    df['ttc_s'] = (np.sign(df['abs_v0']-df['abs_v1'])+1)*0.5*df['ttc_1'] + (np.sign(df['abs_v1']-df['abs_v0'])+1)*0.5*df['ttc_0']
    df['ttc_sum'] = df['ttc_0']+df['ttc_1']
    df['x0_closer'] = np.sign(df.abs_x1 -df.abs_x0  )
    df['x0_first'] = np.sign(df.ttcdiff)                   
    df['end_closer_first'] = abs(df.x0_closer + df.x0_first)*0.5
    df['x0_faster'] = np.sign(df.abs_v0 -df.abs_v1  )
    df['faster_first'] =  abs(df.x0_faster + df.x0_first)*0.5
    df['end_closer_xenddif'] = (df.end_closer_first-0.5)*2 * df.xenddif
    df['first_deltav'] = (df.faster_first-0.5)*2 * df.delta_v
    df['first_xenddif'] = (df.end_closer_first-0.5)*2 * df.xenddif
    df['first_deltav_norm'] = scaler.fit_transform(df[['first_deltav']])
    df['first_xenddif_norm'] = scaler.fit_transform(df[['first_xenddif']])
    
    return(df)

orig_df = pd.read_csv('taupelidata_corners_pilotit.csv')
df = corners_preprocess(orig_df)

    
    