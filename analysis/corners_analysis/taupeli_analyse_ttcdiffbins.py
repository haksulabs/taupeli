#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:04:36 2022

@author: t
"""


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt



#df = pd.read_csv('taupelidata_pilotit.csv')
#df = pd.read_csv('taupelidata_noise.csv')
#df = pd.read_csv('taupelidata_randomloc.csv')
df = pd.read_csv('taupelidata_corners_pilotit.csv')

df['bins'] = pd.cut(df['ttcdiff'], bins=9)
df['binmid'] = df['bins'].apply(lambda x: x.mid)


df['bins'].value_counts()


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
    
    
    
    
