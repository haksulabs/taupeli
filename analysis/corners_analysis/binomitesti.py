# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:51:48 2023

@author: t
"""

import pandas as pd
from scipy.stats import binomtest
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df



# stag1=1 and 0  condditions for updown and leftright
df = orig_df[(orig_df['stag1']==1)]

# Calculate overall average accuracy

overall_success_rate = df['correct'].mean()

conditions = ['d_updown','d_leftright']
results={}
for c in conditions:
    cond_df = df[df[c]==1]
    success = cond_df['correct'].sum()
    trials = cond_df.shape[0]
    results[c] = binomtest(success, trials, p=overall_success_rate,alternative='greater')
    
for k in results.items():
    print(k)



