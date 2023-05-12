# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:07:29 2023

plot accuracies in rolling time window

@author: t
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
df = corners_preprocess(df)

# Create an empty list to store the rolling mean accuracies
rolling_accuracy = []

# Iterate through the unique time values in the DataFrame

dt = 0.05

con = 34
cons = [23, 41]

#cons = [12,34]
stagger = 0 

df_ac = pd.DataFrame(columns=['time'])
dt = 0.1
step = 0.03

stag = 0
for t in np.arange(30):
    time = step*t
    # Find the time values within the specified time interval
    interval_values = df.loc[(df['abs_ttcdiff'] >= time) & (df['abs_ttcdiff'] < time+dt) & (df['condition'].isin(cons)) & (df['stagger']==stag), 'correct']

    # Calculate the rolling mean accuracy for the time interval
    interval_mean = interval_values.mean()
    
    # Append the rolling mean accuracy to the list
    rolling_accuracy.append(interval_mean)
    conname = str(con) + '-' + str(stag)
    d = {'time': time+dt*0.5, conname: interval_mean}
    df_d = pd.DataFrame([d])
    df_ac = pd.concat([df_ac,df_d],ignore_index=True)
        
df_ac0 = df_ac

stag = 1
df_ac = pd.DataFrame(columns=['time'])
for t in np.arange(30):
    time = step*t
    # Find the time values within the specified time interval
    interval_values = df.loc[(df['abs_ttcdiff'] >= time) & (df['abs_ttcdiff'] < time+dt) & (df['condition'].isin(cons)) & (df['stagger']==stag), 'correct']

    # Calculate the rolling mean accuracy for the time interval
    interval_mean = interval_values.mean()

    # Append the rolling mean accuracy to the list
    rolling_accuracy.append(interval_mean)
    conname1 = str(con) + '-' + str(stag)
    d = {'time': time + dt*0.5, conname1: interval_mean}
    df_d = pd.DataFrame([d])
    df_ac = pd.concat([df_ac,df_d],ignore_index=True)

        
df_ac0.insert(2, conname1, df_ac[conname1], True)

# Create a new DataFrame to store the rolling mean accuracies and the corresponding time values
#rolling_df = pd.DataFrame({'abs_ttcdiff': df['time'].unique(), 'rolling_accuracy': rolling_accuracy})

# Plot the rolling accuracy over time
df = df_ac0
plt.plot(df['time'], df[conname], label=conname)
plt.plot(df['time'], df[conname1], label=conname1)
plt.legend()
plt.xlabel('TTC difference')
plt.ylabel('Accuracy')
plt.title(f'Moving Average Accuracy (Window Length = {dt})')
plt.show()



# Set the degree of the polynomial trendline
degree = 2

# Fit a polynomial of degree 'degree' to x1 and x2
p1 = np.polyfit(df['time'], df[conname], degree)
p2 = np.polyfit(df['time'], df[conname1], degree)

# Generate x-values for plotting the trendline
trendline_time = np.linspace(df['time'].min(), df['time'].max(), 100)

# Evaluate the polynomial at the x-values to get the trendline values
trendline_x1 = np.polyval(p1, trendline_time)
trendline_x2 = np.polyval(p2, trendline_time)

# Plot both x1 and x2 with respect to time, along with the trendlines
#plt.plot(df['time'], df[conname], 'o', label='0')
#plt.plot(df['time'], df[conname1], 'o', label='1')
plt.plot(trendline_time, trendline_x1, '-', label='Equal initial distance')
plt.plot(trendline_time, trendline_x2, '-', label='Asymmetric initial distance')

# Add legend, x-label, y-label, and title
plt.ylim(0.5, 1)
plt.legend()
plt.xlabel('TTC difference (s)')
plt.ylabel('Average accuracy')
plt.title(f'Average accuracy in relation to actual TTC difference')

plt.savefig('accuracy_ttcdiff_unilateral.svg', format='svg')

# Show the plot
plt.show()