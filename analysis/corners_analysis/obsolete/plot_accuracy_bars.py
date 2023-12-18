# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 02:16:59 2023

plot mean bars with individual points

reads accuracy_all.csv that was creates by analysis_corners_basic


@author: t
"""


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Load some example data
#tips = sns.load_dataset("tips")

#acu = pd.read_csv('accuracy_all.csv')

# read accuracies from files and reformat lazily

acu1 = pd.read_csv('accuracy_same.csv')
acu2 = pd.read_csv('accuracy_differ.csv')
acu1['startpos']=0
acu2['startpos']=1

#remove averages, we do them here
acu = pd.concat([acu1,acu2])
acu = acu[acu.name != 'total']
acu = acu[acu.condition != 'all']



#Plot the chart:

# Draw the bar chart
ax = sns.barplot(
    data=acu, 
    x="condition", 
    y="accuracy", 
    hue="startpos", 
    alpha=0.7, 
    errorbar=('ci', 95)
)
ax.set_ylim(0.4, 1)

# Get the legend from just the bar chart
handles, labels = ax.get_legend_handles_labels()

# Draw the stripplot
sns.stripplot(
    data=acu, 
    x="condition", 
    y="accuracy", 
    hue="startpos", 
    dodge=True, 
    edgecolor="black", 
    linewidth=.75,
    ax=ax,
)
# Remove the old legend
ax.legend_.remove()
# Add just the bar chart legend back
ax.legend(
    handles,
    labels,
    loc=7,
    bbox_to_anchor=(1.25, .5),
)




