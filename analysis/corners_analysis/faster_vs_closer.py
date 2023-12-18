# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:00:10 2023

examine speed vs time 

# subset of trials with faster object starting farther away
#
# if bias is to select faster object, then accuracy higher 
# when faster overtakes
#
# if bias is to select closer object (not faster), then
# accuracy should be higher when the faster object does not overtake


@author: t
"""
import os
# os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'
if(os.name=='nt'):
    os.environ['R_HOME'] = 'h:/anaconda3/envs/pymer4/Lib/R'
    savedir = 'H:/repoman/taupeli/illustration'
elif(os.name=='posix'):
    os.environ['R_HOME'] = '/usr/t/anaconda3/envs/pymer4/Lib/R'
       
    
import pandas as pd
from scipy.stats import binomtest
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocess_taupelidata import corners_preprocess


df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df.copy()


conditions = ['d_updown', 'd_leftright', 'd_diagonal']

overtake = df[(df['faster_overtake']==1) | (df['faster_noovertake']==1) ].groupby(['name','comp_cond','faster_overtake'])['correct'].agg(['mean','count'])
grouped_data = overtake.groupby(['comp_cond', 'faster_overtake'])['mean'].mean().unstack()

desired_order = ['updown','leftright','diagonal']
grouped_data = grouped_data.reindex(desired_order)
subcondition_names = ['Closer wins','Faster wins']
fig, ax = plt.subplots(figsize=(6, 4))
# Create a bar graph
colors = ['black', 'gray']
edge_color = 'black'
grouped_data.plot(kind='bar', color=colors, edgecolor = edge_color, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# Set titles and labels
ax.set_title('Accuracy in trials when faster object starts farther',fontsize=16)
ax.set_xlabel('Condition',fontsize=15)
ax.set_ylabel('Accuracy',fontsize=15)
ax.set_ylim([0.5, 1])
ax.legend(subcondition_names, loc='upper right', bbox_to_anchor=(1, 1),fontsize = 11)


plt.savefig(savedir+'/'+ 'faster_vs_closer.eps', format='eps', dpi=300)
plt.savefig(savedir+'/'+ 'faster_vs_closer.png', format='png', dpi=300)