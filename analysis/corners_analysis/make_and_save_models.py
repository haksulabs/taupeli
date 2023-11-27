# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:10:54 2023

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
from pymer4.utils import get_resource_path
from pymer4.models import Lm
from pymer4.models import Lmer
from pymer4.io import save_model, load_model


from preprocess_taupelidata import corners_preprocess

def summary_to_df(model,oldsummary_df):
    # get some data from the model and collect them in df
    varnames = ['model_name', 'formula','AIC']
    model_data = {varname: getattr(model, varname) for varname in varnames} 
    model.coefs
    model_df = pd.DataFrame([model_data])
    newsummary_df = pd.concat([oldsummary_df, model_df], ignore_index=True)    
    return newsummary_df


df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df


formulas = {
    'baseline': 'correct ~  0 + abs_ttcdiff + (0 + abs_ttcdiff|name)',
    'stagger_signed_combcond' : 'correct ~ 0 + abs_ttcdiff + stag1:d_updown +stag1:d_leftright + stag1:d_diagonal + (0 + abs_ttcdiff|name) ',
    'stagger_signed_combcond_randstag' : 'correct ~ 0 + abs_ttcdiff + stag1:d_updown +stag1:d_leftright + stag1:d_diagonal + (0 + abs_ttcdiff +stag1|name) ',
    'delta_xf_end_signed_combcond' : 'correct ~ 0 + abs_ttcdiff + delta_xf_end:d_updown +delta_xf_end:d_leftright + delta_xf_end:d_diagonal +  (0 + abs_ttcdiff + delta_xf_end|name)',
    'stagger_vs_delta_xf' : 'correct ~ 0 + abs_ttcdiff + stag1:d_updown + stag1:d_leftright + stag1:d_diagonal + delta_xf_end:d_updown +delta_xf_end:d_leftright + delta_xf_end:d_diagonal + (0 + abs_ttcdiff|name) '
    }

models ={}
summary_df = pd.DataFrame()
for k, v in formulas.items():
    current_filename = 'model_' + k + '.joblib'
    try:
        current_model = load_model('model_baseline.joblib')
        exists = 1
    except:
        exists = 0
    exists = 0
    if( not exists): 
        current_model = Lmer(v, data=df, family = 'binomial')
        current_model.fit()
        current_model.model_name = k
        save_model(current_model,current_filename)
    summary_df = summary_to_df(current_model,summary_df)
    models[k] = current_model
    


    
#model_baseline = Lmer("correct ~  0 + abs_ttcdiff + (0 + abs_ttcdiff|name) ", data=df, family = 'binomial')
#model_baseline.fit()

#summary_df = pd.DataFrame()

#summary_df = summary_to_df(model_baseline,summary_df)


