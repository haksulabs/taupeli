#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:13:54 2023

@author: t
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:08:58 2023

@author: t
"""

import os
# os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'
if(os.name=='nt'):
    os.environ['R_HOME'] = 'h:/anaconda3/envs/pymer4/Lib/R'
    modelsavepath = 'G:/cachet/taupeli_models/'
elif(os.name=='posix'):
    os.environ['R_HOME'] = '/Users/t/opt/anaconda3/envs/pymer4/lib/R'
    modelsavepath = '/Users/t/cachet/'
         
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lm
from pymer4.models import Lmer
from pymer4.io import save_model, load_model

from sklearn.preprocessing import StandardScaler
from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df

#model_baseline = Lmer("correct ~    abs_ttcdiff + 0  + (0 + abs_ttcdiff|name) ", data=df, family = 'binomial')
#model_baseline.fit()

#model_baseline_intercept = Lmer("correct ~    abs_ttcdiff + 1  + (1 + abs_ttcdiff|name) ", data=df, family = 'binomial')
#model_baseline_intercept.fit()


#model = load_model(modelsavepath + 'model_stagger_signed_combcond_randstag.joblib')
#model = load_model(modelsavepath + 'model_stagger_cat.joblib' )

model = load_model(modelsavepath + 'model_delta_fs_end.joblib' )

# predict dataset 

#stags = [-1, 0, 1]
stags = np.percentile(df['delta_fs_end'],[5,50,95])
stags[1]=0

cond1 = {'d_updown' : 1, 'd_leftright' : 0, 'd_diagonal' : 0}
cond2 = {'d_updown' : 0, 'd_leftright' : 1, 'd_diagonal' : 0}
cond3 = {'d_updown' : 0, 'd_leftright' : 0, 'd_diagonal' : 1}
conds = [cond1,cond2,cond3]


predict_df=pd.DataFrame()
# for s in stags:
#     for c in conds:        
#         x_values = np.linspace(df['abs_ttcdiff'].min(), df['abs_ttcdiff'].max(), 100)
#         pre_df = pd.DataFrame({'abs_ttcdiff': x_values, 'stag1':s, 'd_updown':c['d_updown'] , 'd_leftright':c['d_leftright'], 'd_diagonal': c['d_diagonal'] })
#         predict_df = pd.concat([predict_df,pre_df], ignore_index=True)




for s in stags:
    for c in conds:        
        x_values = np.linspace(df['abs_ttcdiff'].min(), df['abs_ttcdiff'].max(), 100)
        pre_df = pd.DataFrame({'abs_ttcdiff': x_values, 'stag1_1':int(s==1),'stag1_0':int(s==0), 'stag1__1':int(s==-1)  , 'stag1_cat':str(s), 'stag1':s,'delta_fs_end':s , 'd_updown':c['d_updown'] , 'd_leftright':c['d_leftright'], 'd_diagonal': c['d_diagonal'] })
        predicted_accuracies = pd.DataFrame(model.predict(pre_df, use_rfx=False, verify_predictions=False, verbose=True, skip_data_checks=True))

        plt.plot(pre_df['abs_ttcdiff'], predicted_accuracies, label=(s, c))
        
plt.legend()    
plt.show()
  
    # predict_df = pd.concat([predict_df,pre_df], ignore_index=True)                        
        
        
        
        
        
        
# # interactions
# predict_df['stag1:d_updown'] = predict_df['stag1'] *  predict_df['d_updown']
# predict_df['stag1:d_leftright'] = predict_df['stag1'] *  predict_df['d_leftright']
# predict_df['stag1:d_diagonal'] = predict_df['stag1'] *  predict_df['d_diagonal']

# predicted_accuracies = pd.DataFrame(model.predict(predict_df, use_rfx=False, verify_predictions=False, verbose=True, skip_data_checks=True))

# predict_df['predicted_accuracy'] = predicted_accuracies


# # Plotting

# # Filtering data based on conditions
# updown_data = predict_df[  (predict_df['d_updown'] == 1) & (predict_df['stag1'] == 0)]
# updown_stag_data = predict_df[  (predict_df['d_updown'] == 1) & (predict_df['stag1'] == -1 )]
# updown_stag1_data = predict_df[  (predict_df['d_updown'] == 1) & (predict_df['stag1'] == 1 )]



# leftright_data = predict_df[  (predict_df['d_leftright'] == 1) & (predict_df['stag1'] == 0)]
# leftright_stag_data = predict_df[  (predict_df['d_leftright'] == 1) & (predict_df['stag1'] == -1)]
# leftright_stag1_data = predict_df[  (predict_df['d_leftright'] == 1) & (predict_df['stag1'] == 1)]

# diagonal_data = predict_df[  (predict_df['d_diagonal'] == 1) & (predict_df['stag1'] == 0)]
# diagonal_stag_data = predict_df[  (predict_df['d_diagonal'] == 1) & (predict_df['stag1'] == -1)]
# diagonal_stag1_data = predict_df[  (predict_df['d_diagonal'] == 1) & (predict_df['stag1'] == 1)]

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(updown_data['abs_ttcdiff'], updown_data['predicted_accuracy'], label='Up/Down stag = 0')
# plt.plot(updown_stag_data['abs_ttcdiff'], updown_stag_data['predicted_accuracy'],  label='Up/Down stag=-1')
# plt.plot(updown_stag1_data['abs_ttcdiff'], updown_stag1_data['predicted_accuracy'],  label='Up/Down stag=1')


# plt.plot(leftright_data['abs_ttcdiff'], leftright_data['predicted_accuracy'], linestyle='--', label='Left/Right stag = 0')
# plt.plot(leftright_stag_data['abs_ttcdiff'], leftright_stag_data['predicted_accuracy'], linestyle='--', label='Left/Right stag=-1')
# plt.plot(leftright_stag1_data['abs_ttcdiff'], leftright_stag1_data['predicted_accuracy'], linestyle='--', label='Left/Right stag=1')


# plt.plot(diagonal_data['abs_ttcdiff'], diagonal_data['predicted_accuracy'], linestyle=':', label='Diagonal stag = 0')
# plt.plot(diagonal_stag_data['abs_ttcdiff'], diagonal_stag_data['predicted_accuracy'], linestyle=':', label='Diagonal stag=-1')
# plt.plot(diagonal_stag1_data['abs_ttcdiff'], diagonal_stag1_data['predicted_accuracy'], linestyle=':', label='Diagonal stag=1')



# #plt.plot(leftright_data['abs_ttcdiff'], leftright_data['predicted_accuracy'], marker='s', label='Left/Right')

# # plt.plot(diagonal_data['abs_ttcdiff'], diagonal_data['predicted_accuracy'], marker='^', label='Diagonal')

# # 
# # plt.plot(leftright_data['abs_ttcdiff'], leftright_data['predicted_accuracy'], marker='s', label='Left/Right stag1')
# # plt.plot(diagonal_data['abs_ttcdiff'], diagonal_data['predicted_accuracy'], marker='^', label='Diagonal stag1')


# plt.xlabel('abs_ttcdiff')
# plt.ylabel('Predicted Accuracy')
# plt.title('Predicted Accuracy vs abs_ttcdiff for Different Conditions')
# plt.legend()
# plt.grid(True)
# plt.show()

