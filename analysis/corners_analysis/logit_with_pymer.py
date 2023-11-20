# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:21:33 2023

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
from pymer4.utils import get_resource_path
from pymer4.models import Lm
from pymer4.models import Lmer

from sklearn.preprocessing import StandardScaler
from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df

# con=23
# print('condition ', con)
# df =orig_df[orig_df['condition']==con]

# model = Lm("correct ~ 1 +  abs_ttcdiff_norm", data=df, family = 'binomial')
# print(model.fit())

# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df, family = 'binomial')
# print(model.fit())



# print('startpos 11')
# df1 = df[df['startpos'].isin([11,55])]
# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df1, family = 'binomial')
# print(model.fit())




# print('startpos 15')
# df2 = df[df['startpos'].isin([15,51])]
# model = Lm("correct ~ 1 +  abs_ttcdiff_norm + xenddif_norm", data=df2, family = 'binomial')
# print(model.fit())

# # condition 12
# print('condition 23')
# df =orig_df[orig_df['condition']==23]



# model = Lm("correct ~ 1 +  abs_ttcdiff", data=df, family = 'binomial')
# print(model.fit())

# model = Lm("correct ~ 1 +  abs_ttcdiff + xenddif", data=df, family = 'binomial')
# print(model.fit())







# multi level models

# calculate correlations

# con = 12
# stag = 0
# df =df[df['stagger']==stag]
# df =orig_df[orig_df['condition'].isin([con])]
# corr_matrix_absolutevalues = df[['abs_ttcdiff_norm', 'delta_x_end_norm', 'delta_v_norm','tot_separation_norm']].corr()
# corr_matrix = df[['abs_ttcdiff_norm', 'delta_xf_end', 'delta_vf','tot_separation_norm']].corr()




conditions = [12,34,23,41,24,13]
preds = ['abs_ttcdiff_norm','tot_separation_norm','delta_xf_end_norm', 'delta_vf_norm','stagger','n_trials','reply_ttc_time','min_v','max_v']



new_columns = conditions.copy()
for con in conditions:   
    new_columns.append(f"{con}s")
dfres_simple = pd.DataFrame(index=preds, columns=new_columns)

# simple regression models 

    
for con in conditions: 
    

    df =orig_df[orig_df['condition'].isin([con])]  
    print('condition:', con)    
    
   
    for pred in preds:
        modelpara = "correct ~ " + pred + "  + (1|name) "        
        model = Lmer(modelpara, data=df, family = 'binomial')
        print(model.fit())
        tempres_df = model.coefs.Estimate
        tempres_sig = model.coefs.Sig
        
        sigloc = f"{con}s"
        
        dfres_simple.loc[pred, con] = tempres_df.loc[pred]
        dfres_simple.loc[pred,sigloc] = tempres_sig[pred]
        #dfres_simple[ns] = results_sig
    

# for con in conditions: 
#     stag = 1
    
#     #df =orig_df[orig_df['condition'].isin(all)]
#     df =orig_df[orig_df['condition'].isin([con])]
#     df =df[df['stagger']==stag]            
    
#     print('condition:', con)
#     #model = Lmer("correct ~   abs_ttcdiff_norm  +end_closer_xenddif + faster_first +  (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~    abs_ttcdiff_norm + tot_separation_norm + first_deltav + end_closer_xenddif + (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~    abs_ttcdiff_norm + tot_separation_norm + first_xenddif_norm + first_deltav_norm + (1|name) ", data=df, family = 'binomial')
#     model = Lmer("correct ~    tot_separation_norm + delta_x_end_norm + delta_v_norm + end_ closer_first+ (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~   abs_ttcdiff_norm +  (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~   tot_separation_norm +  (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~   v_div_v_norm +  (1|name) ", data=df, family = 'binomial')
#     #model = Lmer("correct ~  tot_separation_norm +  delta_x_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
#     #model = Lm("correct ~ abs_ttcdiff_norm+ tot_separation_norm +  totdif_norm +  delta_v_norm ", data=df, family = 'binomial')
#     print(model.fit())


dfres=pd.DataFrame(columns=conditions)
    
for con in conditions: 
    
    # calc with respect to the faster, without taking absolute values
    
    #stag = 1
    
    #df =orig_df[orig_df['condition'].isin(all)]
    df =orig_df[orig_df['condition'].isin([con])]
    #df =df[df['stagger']==stag]       

    #df = df[df['overtake_oclusion']==1]
    
    print('condition:', con)
    # absolute values
    
 #   model = Lmer("correct ~   tot_separation_norm + delta_x_end_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
    #model = Lmer("correct ~   abs_ttcdiff_norm + tot_separation_norm + delta_x_end_norm + delta_v_norm + (abs_ttcdiff_norm|name) ", data=df, family = 'binomial')
    # sign matters
    model = Lmer("correct ~    abs_ttcdiff_norm + tot_separation_norm + abs_ttcdiff_norm + delta_xf_end_norm + delta_vf_norm + stagger + (1|name) ", data=df, family = 'binomial')

    #model = Lmer("correct ~   abs_ttcdiff_norm +  (1|name) ", data=df, family = 'binomial')
    #model = Lmer("correct ~   tot_separation_norm +  (1|name) ", data=df, family = 'binomial')
    #model = Lmer("correct ~   v_div_v_norm +  (1|name) ", data=df, family = 'binomial')
    #model = Lmer("correct ~  tot_separation_norm +  delta_x_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
    #model = Lm("correct ~ abs_ttcdiff_norm+ tot_separation_norm +  totdif_norm +  delta_v_norm ", data=df, family = 'binomial')
    print(model.fit())
    
    results_for_con = model.coefs.Estimate
    results_sig = model.coefs.Sig
    
    ns = str(con)+" sig"
    dfres[con] = results_for_con
    dfres[ns] = results_sig
        
       # calc with respect to the faster, without taking absolute values
  
    
# use dummy variables for direction
     
#model = Lmer("correct ~    tot_separation_norm + delta_xf_end + delta_vf + d_12.0 + d_13.0 +  d_41.0 + d_23.0 + d_24.0 + d_34.0 + (1|name) ", data=df, family = 'binomial')
#print(model.fit())

    
    
    
# con = [12]
# stag = 0
# df =orig_df[orig_df['condition'].isin(con)]
# df =df[df['stagger']==stag]            
# dfcon = df.copy()

# print('--------------------------------------------')
# print('multilevel models')
# print('condition:', con)

#print('basic model with only ttcdiff')
#model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
#print(model.fit())

#print('model ttcdiff  + yseparation')

# model_lm = Lm("correct ~  abs_ttcdiff_norm ", data=df, family = 'binomial')
# print(model_lm.fit())

#model = Lmer("correct ~  abs_ttcdiff_norm + ttc_f + ttc_s + (1|name) ", data=df, family = 'binomial')
# model = Lmer("correct ~   abs_ttcdiff_norm + tot_separation_norm * v_div_v_norm +  (1|name) ", data=df, family = 'binomial')
#model = Lmer("correct ~  tot_separation_norm +  delta_x_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
#model = Lm("correct ~ abs_ttcdiff_norm+ tot_separation_norm +  totdif_norm +  delta_v_norm ", data=df, family = 'binomial')
# print(model.fit())





#model = Lmer("correct ~  abs_ttcdiff_norm + tot_separation_norm + delta_v_norm +  (1|name) ", data=df, family = 'binomial')
#model = Lmer("correct ~  tot_separation_norm +  delta_x_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
#model = Lm("correct ~ abs_ttcdiff_norm+ tot_separation_norm +  totdif_norm +  delta_v_norm ", data=df, family = 'binomial')
#print(model.fit())

#model2 = Lmer("correct ~ abs_ttcdiff_norm + sum_v +    (1|name) ", data=df, family = 'binomial')
#print(model2.fit())

# res_df = pd.DataFrame()
# pred_candidates = ['delta_v_norm','delta_x_norm','x_div_x','v_div_v','sum_v','tot_separation_norm']
# for pred in pred_candidates:
#     df['predictor'] = df[pred]
#     model = Lmer("correct ~ abs_ttcdiff_norm + predictor + (1|name) ", data=df, family = 'binomial')
#     model.fit()
#     print(pred + ':' + str(model.AIC))
#     d = {'predictor': pred, 'AIC': model.AIC}
#     d_df = pd.DataFrame([d])
#     res_df =  pd.concat([res_df, d_df], ignore_index=True)
    
    
# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')


############

# print('startpos 11 nostagger')
# df = dfcon[dfcon['startpos'].isin([11,55])]

# print('basic model with only ttcdiff')
# model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')




# ###########
# print('startpos 15 stagger')


# df = dfcon[dfcon['startpos'].isin([15,51])]

# print('basic model with only ttcdiff')
# model = Lmer("correct ~ abs_ttcdiff_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yenddif_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yenddif_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('model ttcdiff  + yseparation + yenddif + delta_v_norm')
# model = Lmer("correct ~ abs_ttcdiff_norm + yseparation_norm + yenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())






# con = 12
# df =orig_df[orig_df['condition']==con]
# dfcon = df.copy()

# print('------con 12--------------------------------------')
# print('nostagger')
# df = dfcon[dfcon['startpos'].isin([11,55])]

# model = Lmer("correct ~ abs_ttcdiff_norm + xseparation_norm + xenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())

# print('############')

# print('stagger')

# df = dfcon[dfcon['startpos'].isin([15,51])]

# model = Lmer("correct ~ abs_ttcdiff_norm + xseparation_norm + xenddif_norm + delta_v_norm + (1|name) ", data=df, family = 'binomial')
# print(model.fit())







