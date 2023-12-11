# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:00:10 2023

examine speed vs time 

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
df = orig_df.copy()

# subset of trials with faster object starting farther away
#
# if bias is to select faster object, then accuracy higher 
# when faster overtakes
#
# if bias is to select closer object (not faster), then
# accuracy should be higher when the faster object does not overtake



conditions = ['d_updown', 'd_leftright', 'd_diagonal']

res = {}
for c in conditions: 
    df = orig_df.copy()
    df=df[df[c]==1]

    faster_overtake_correct = df[(df['faster_overtake']==1)].correct.mean()
    faster_noovertake_correct = df[(df['faster_noovertake']==1)].correct.mean()
    res[c] = [faster_overtake_correct, faster_noovertake_correct]



plt.figure()
plt.bar(['Updown\nFaster\novertakes', 'Updown\nFaster\nno overtake', 
         'Leftright\nFaster\novertakes', 'Leftright\nFaster\nno overtake',
         'Diagonal\nFaster\novertakes', 'Diagonal\nFaster\nno overtake' ], 
        [ res['d_updown'][0], res['d_updown'][1],
          res['d_leftright'][0], res['d_leftright'][1],
          res['d_diagonal'][0], res['d_diagonal'][1] ] )
        
         
plt.ylabel('Percentage')
plt.title('Percentage of Correct Trials in Two DataFrames')
plt.ylim(0, 1)  # Setting y-axis limit to 100 for percentage view
plt.show()