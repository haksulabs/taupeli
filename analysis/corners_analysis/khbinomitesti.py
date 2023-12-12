#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:18:20 2023

@author: t
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:51:48 2023

@author: t
"""

import pandas as pd
from scipy.stats import binomtest
from scipy.stats import ttest_rel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df

df = df[df['name'].str.contains('kh')]

df = df[df['stag1']==-1]

df['merged_condition']=''
#df['merged_condition'] = ('updown' * #df['d_updown']) + ('leftright' * df['d_leftrigt'])

df['merged_condition'][df.d_updown != 0] = 'updown'
df['merged_condition'][df.d_leftright != 0] = 'leftright'
df['merged_condition'][df.d_diagonal != 0] = 'diagonal'

df_means = df.groupby(['name','merged_condition']).mean()
df_means = df_means.reset_index()



grouped_df = df.groupby(['name', 'merged_condition'])['correct'].mean().reset_index()

# Pivot the DataFrame
pivot_df = grouped_df.pivot(index='name', columns='merged_condition', values='correct')

# Assuming your conditions are 'condition1' and 'condition2'
# Calculate the difference between these conditions
pivot_df['difference'] = pivot_df['updown'] - pivot_df['leftright']

# Reset the index to make 'name' a column again
final_df = pivot_df.reset_index()
# stag1=1 and 0  condditions for updown and leftright

success_dif = (pivot_df.difference>0).sum()


result = binomtest(success_dif, len(pivot_df))



t_statistic, p_value = ttest_rel(pivot_df['updown'], pivot_df['leftright'])

