#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:26:42 2023

@author: t
"""

"""
Created on Sun Feb 19 08:21:33 2023

multilevel models, all con in same models 

taupeli logit with pymer


run twice if first run produces dll error 

@author: t
"""
import os
# os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'
if(os.name=='nt'):
    os.environ['R_HOME'] = 'h:/anaconda3/envs/pymer4/Lib/R'
elif(os.name=='posix'):
    os.environ['R_HOME'] = '/usr/t/anaconda3/envs/pymer4/Lib/R'
    
         
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pymer4.models import Lm
from pymer4.models import Lmer


from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df.copy()



preds = ['abs_ttcdiff_norm','tot_separation_norm','delta_xf_end_norm', 'delta_vf_norm','stagger','n_trials','reply_ttc_time','min_v','max_v']

# model_baseline = Lmer("correct ~    abs_ttcdiff + 0  + (abs_ttcdiff + 0|name) ", data=df, family = 'binomial')
# model_baseline.fit()


# model_stagger_binary = Lmer("correct ~    abs_ttcdiff + 0 + stagger + (abs_ttcdiff + 0 |name) ", data=df, family = 'binomial')
# model_stagger_binary.fit()

# model_stagger_signed = Lmer("correct ~    abs_ttcdiff + 0 + stag1 + (abs_ttcdiff + 0 +stag1|name) ", data=df, family = 'binomial')
# model_stagger_signed.fit()

# model_stagger_signed_cond = Lmer("correct ~    abs_ttcdiff + 0  + stag1:d_12 +stag1:d_34 + stag1:d_23 + stag1:d_41 + stag1:d_24  + stag1:d_13 + (abs_ttcdiff + 0 +stag1|name) ", data=df, family = 'binomial')
# model_stagger_signed_cond.fit()
# print(model_stagger_signed_cond.summary())

model_stagger_signed_combcond1 = Lmer("correct ~    abs_ttcdiff + stag1:d_updown +stag1:d_leftright + stag1:d_diagonal +  (0 + abs_ttcdiff +stag1|name) ", data=df, family = 'binomial')
model_stagger_signed_combcond1.fit()
print(model_stagger_signed_combcond1.summary())


model_stagger_signed_combcond2 = Lmer("correct ~    abs_ttcdiff + delta_xf_end:d_updown +delta_xf_end:d_leftright + delta_xf_end:d_diagonal +  (0 + abs_ttcdiff + delta_xf_end|name) ", data=df, family = 'binomial')
model_stagger_signed_combcond2.fit()
print(model_stagger_signed_combcond2.summary())




# kysymys:  onko alku lähempi  enempi vai vähempi tärkee kuin lopussa lähempi

# model_stagstart_vs_end1 = Lmer("correct ~  0 + abs_ttcdiff  +stag1 + (0 + abs_ttcdiff+stag1|name) ", data=df, family = 'binomial')
# model_stagstart_vs_end1.fit()
# print(model_stagstart_vs_end1.summary())

# model_stagstart_vs_end2 = Lmer("correct ~  0 + abs_ttcdiff  +delta_xf_end + (0 + abs_ttcdiff+delta_xf_end|name) ", data=df, family = 'binomial')
# model_stagstart_vs_end2.fit()
# print(model_stagstart_vs_end2.summary())
# xenddif better by aic -700











