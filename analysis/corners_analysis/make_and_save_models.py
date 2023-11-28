# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:10:54 2023

@author: t
"""

import os
# os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'

modelsavepath=''
if(os.name=='nt'):
    os.environ['R_HOME'] = 'h:/anaconda3/envs/pymer4/Lib/R'    
    modelsavepath = 'G:/cachet/taupeli_models/'

elif(os.name=='posix'):
    os.environ['R_HOME'] = '/Users/t/opt/anaconda3/envs/pymer4/lib/R'
    modelsavepath = '/users/t/cachet/'
    
         
 
import numpy as np
import pandas as pd
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
    'stagger_vs_delta_xf' : 'correct ~ 0 + abs_ttcdiff + stag1:d_updown + stag1:d_leftright + stag1:d_diagonal + delta_xf_end:d_updown +delta_xf_end:d_leftright + delta_xf_end:d_diagonal + (0 + abs_ttcdiff|name) ',
    'delta_x_mean_signed_combcond' : 'correct ~ 0 + abs_ttcdiff + delta_x_mean:d_updown +delta_x_mean:d_leftright + delta_x_mean:d_diagonal +  (0 + abs_ttcdiff + delta_x_mean|name)',
    'more_fixed_effects' : 'correct ~ 0 + abs_ttcdiff + tot_separation + delta_xf_end + delta_vf +  stag1:d_updown +stag1:d_leftright + stag1:d_diagonal + (0 + abs_ttcdiff +stag1|name)',
    'even_more_fixed_effects' : 'correct ~ 0 + abs_ttcdiff + tot_separation + delta_x_mean + delta_xf_end + delta_vf + abs_vf + abs_vs +  stag1:d_updown +stag1:d_leftright + stag1:d_diagonal + (0 + abs_ttcdiff +stag1|name)'
    }

models ={}
summary_df = pd.DataFrame()
for k, v in formulas.items():
    current_filename = modelsavepath + 'model_' + k + '.joblib'
    try:
        current_model = load_model(current_filename)
        exists = 1
        
        #liimaa: tarkista että malli on sama.. 
        test_model = Lmer(v, data=df, family = 'binomial')
        if(test_model.formula != current_model.formula):
            print('model changed, fitting new: ' + k)
            exists=0
    except:
        exists = 0
        
    if( not exists): 
        current_model = Lmer(v, data=df, family = 'binomial')
        current_model.fit()
        current_model.model_name = k
        save_model(current_model,current_filename)
    summary_df = summary_to_df(current_model,summary_df)
    models[k] = current_model
    

