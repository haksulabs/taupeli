# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:46:02 2023

@author: t
"""

import os
# os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'
if(os.name=='nt'):
    os.environ['R_HOME'] = 'h:/anaconda3/envs/pymer4/Lib/R'
    savedir = 'H:/repoman/taupeli/illustration'
elif(os.name=='posix'):
    os.environ['R_HOME'] = '/usr/t/anaconda3/envs/pymer4/Lib/R'
            

 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats

from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df


# calculate subject means and confidence intervals
subject_means = df.groupby(['name', 'condition', 'stag1'])['correct'].mean().reset_index()


# Calculate overall mean and SEM for each condition
mean_accuracy = subject_means.groupby(['condition', 'stag1'])['correct'].mean()
sem_accuracy = subject_means.groupby(['condition', 'stag1'])['correct'].sem()


# Calculate the confidence interval (95% CI using t-distribution)
ci = sem_accuracy * stats.t.ppf((1 + 0.95) / 2., len(subject_means['name'].unique()) - 1)
ci = ci.unstack()

accuracy_df = df.groupby(['condition', 'stag1'])['correct'].mean().unstack()
desired_order = [12, 34, 23, 41, 24, 13]
condition_names = ['\\/', '/\\','>','<','\\','/']
accuracy_df = accuracy_df.reindex(desired_order)


# Plotting
fig, ax = plt.subplots(figsize=(6, 4))

colors = ['black', 'white', 'gray']
edge_color = 'black'

subcondition_names = ['Farther distance', 'Equal distance', 'Shorter distance']
accuracy_df.plot(kind='bar', yerr=ci, color = colors, edgecolor=edge_color,  ax=ax)
ax.set_xticklabels(condition_names, rotation=0,fontsize=12)
ax.set_title('Accuracy by condition and initial target position',fontsize=16)
ax.set_xlabel('Condition',fontsize=15)
ax.set_ylabel('Accuracy',fontsize=15)
ax.set_ylim([0.5, 1])
#plt.rc('ytick', labelsize=10)

ax.legend(subcondition_names, loc='upper right', bbox_to_anchor=(1, -0.1),fontsize = 11)
plt.tight_layout()

plt.show()

plt.savefig(savedir+'/'+ 'accuracy_with_ci.eps', format='eps', dpi=300)
plt.savefig(savedir+'/'+ 'accuracy_with_ci.png', format='png', dpi=300)