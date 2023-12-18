# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:21:54 2023

stepwise multiregression analysis of corner data with pymer

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
from scipy.stats import chi2

from preprocess_taupelidata import corners_preprocess


def likelihood_ratio_test(model1, model2):
    """
    Perform a likelihood ratio test between two nested models.
    """

    lr = 2 * (model2.logLike - model1.logLike)
    # assume same random effects, calculate the number of fixed effects:
    degrees_diff = len(model2.coefs) - len(model1.coefs)
    p = chi2.sf(lr, df=degrees_diff)
    return lr, p


df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df


conditions = [12,34,23,41,24,13]
con = 12

df =orig_df[orig_df['condition'].isin([con])]

# Initial model
random_intercepts_formula = '+ (1|name)'
base_model_formula = 'correct ~ 1' 
current_model_formula = base_model_formula

print('base model:')
print(base_model_formula + random_intercepts_formula)


current_model = Lmer(current_model_formula + random_intercepts_formula, data=df, family='binomial')
current_model.fit()


predictors = ['abs_ttcdiff_norm','tot_separation_norm','delta_xf_end_norm', 'delta_vf_norm','stagger','n_trials','reply_ttc_time','min_v','max_v']

# add predictors to model and check if the model improves
included_predictors = []

#while True:
#    changed = False
# Forward Selection

for predictor in predictors:
    if predictor not in included_predictors:
        new_model_formula = current_model_formula + ' + ' + predictor
        print('############')
        print('new model:')
        print(new_model_formula)
        
        new_model = Lmer(new_model_formula + random_intercepts_formula, data=df,family='binomial')
        new_model.fit(silent=True)
        print(new_model.summary())

        lr, p = likelihood_ratio_test(current_model, new_model)
        print('got lr, p')
        print(p)
        if p < 0.05:  # Adjust significance level as needed
            print(f"Adding {predictor} (p={p})")
            current_model = new_model
            current_model_formula = new_model_formula
            included_predictors.append(predictor)
            changed = True
            
            
#marginal_r_squared = results.rvf
#conditional_r_squared = results.rvc            
                #break

#     if not changed:
#         # No more predictors can be added, start Backward Elimination
#         for predictor in included_predictors:
#             new_model_formula = 'correct ~ ' + ' + '.join([p for p in included_predictors if p != predictor])
#             new_model = Lmer(new_model_formula + random_intercepts_formula, data=df, family='binomial')
#             new_model.fit()

#             lr, p = likelihood_ratio_test(current_model, new_model)
#             if p >= 0.05:  # If removing doesn't significantly worsen the model
#                 print(f"Removing {predictor} (p={p})")
#                 current_model = new_model
#                 included_predictors.remove(predictor)
#                 changed = True
#                 break

#     if not changed:
#         # No more changes possible
#         break

# # Final Model
print("Final model:", current_model.formula)





# for predictor in predictors:
#     new_model_part = base_model_part + ' + ' + predictor
#     new_model_formula = new_model_part + model_intercepts
#     print(new_model_formula)
    
    
#     new_model = Lmer(new_model_formula, data=df)
#     new_model.fit()

#     lr, p = likelihood_ratio_test(model, better_model)
#     if p < 0.05:  # Adjust significance level as needed
#         print(f"Adding {predictor} (p={p})")
#         included_predictors.append(predictor)
#         base_model_part = new_model_part
#         base_model = base_model_part + model_intercepts
#         current_model = new_model
#     else:
#         print(f"Excluding {predictor} (p={p})")
        
# # backwards elimination 

# for predictor in included_predictors:
#     new_model_formula = 'correct ~ ' + ' + '.join([p for p in included_predictors if p != predictor]) + model_intercepts
#     new_model = Lmer(new_model_formula, data=df)
#     new_model.fit()

#     lr, p = likelihood_ratio_test(current_model, new_model)
#     if p >= 0.05:  # If removing doesn't significantly worsen the model
#         print(f"Removing {predictor} (p={p})")
#         current_model = new_model
#         included_predictors.remove(predictor)
#         changed = True
#             break
        

        