# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:29:26 2023

@author: t
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:09:12 2022

1-kh trial per trial 
calc likelihoods


@author: t
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit
from preprocess_taupelidata import corners_preprocess


def p_bias_x(x,row):
    # apply bias towards closer object
    bias = x *row.end_closer_xenddif
    ttc_diff_var = 0.55 # optimized value from p_ttcdiff
    p_wrong = scipy.stats.norm(abs(row.ttcdiff)+bias, ttc_diff_var).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)     
    return p1



def p_ttcdiff(x,row):
    # constant variance applied to ttc-diff measurement
    bias = biases[con] *row.end_closer_xenddif
    p_wrong = scipy.stats.norm(abs(row.ttcdiff)+bias, x).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)     
    return p1

def p_ttcs(x,row):
    # variance is sum of observed variances for tau0 and tau1
    bias = biases[con] * row.end_closer_xenddif
    t0 = abs(row.x0_end)/abs(row.v0)
    t1 = abs(row.x1_end)/abs(row.v1)
    sigma0 = x * t0
    sigma1 = x * t1
    
    ttc_diff_var = sigma0**2 + sigma1**2
    
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff+bias, np.sqrt(ttc_diff_var)).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)
    return p1    


def p_vdiv_v(x,row):
    # add noise proportional to relative speeds
    
    sigma0 = x * row.min_v/row.max_v
        
        
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff, sigma0).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)
    return p1    

    

def p_velocity(x,row):
    # calculate variance for ttc-diff based on velocity
    bias = biases[con] * row.end_closer_xenddif
    sigma0 = row.abs_v0*x
    sigma1 = row.abs_v1*x
    
    ttc_diff_var = sigma0**2 + sigma1**2
    
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff+bias, np.sqrt(ttc_diff_var)).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)
    return p1    

def p_eccentricity(x,row,bias=0):
    # calculate variance for ttc-diff based on eccentricity 
    bias = biases[con] * row.end_closer_xenddif
    sigma0 = row.abs_x0*x
    sigma1 = row.abs_x1*x
    
    ttc_diff_var = sigma0**2 + sigma1**2
    
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff+bias, np.sqrt(ttc_diff_var)).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)
    return p1

def p_eccentricity_times_v(x,row):
    # calculate variance for ttc-diff based on eccentricity 
    stdcoeff = x 
    x0 = row.x0
    v0 = row.v0
    x1 = row.x1
    v1 = row.v1
    x0endpos = x0 + v0*0.5
    x1endpos = x1 - v1*0.5
 #   print(x0endpos, x1endpos)
    
    sigma0 = abs(x0endpos)*stdcoeff * abs(v0)
    sigma1 = abs(x1endpos)*stdcoeff * abs(v1)
    
    ttc_diff_var = sigma0**2 + sigma1**2
    
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff, np.sqrt(ttc_diff_var)).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
        print(row.correct)
    return p1

def p_simple(x, row,bias=0):
    if(row.correct==1):
        p1=x
    elif(row.correct==0):
        p1=1-x
    else:
        print('hullu')
        print(row.correct)
    return p1

def m(x,p_func):
    # x = parameter to minimize
    # p_func =  function to calculate probability, given par and row
    
    likelihoods=[]
    for i, row in dfkh.iterrows():       
        # p = probability of user response based on model
        p = p_func(x, row)
        likelihoods.append(p)
        lsum = np.sum(-np.log(likelihoods))
    print('xvalue, likelihoods summa', x, lsum)
    return lsum



orig_df = pd.read_csv('taupelidata_corners_pilotit.csv')
df = corners_preprocess(orig_df)


#name = 'kh27_corners'
#name = 'Samuel'
#name = 'corners_kh44'    

#print(m(1))
guess = [0.2]
#best = minimize(m,guess,args=(p_simple),bounds=((None,0.99),))

#best = minimize(m,guess,args=(p_ttcdiff),bounds=((None,4),))



#best = minimize(m,guess,args=(p_eccentricity),bounds=((None,0.99),))
#best = minimize(m,guess,args=(p_ttcdiff),bounds=((None,3),))


# calculate all conditions separately 
df_res = pd.DataFrame()
conditions = [12,34,23,41,24,13]
stags = [0,1]

# apply bias per condition
biases = {12:1.870759463, 34:2.130667154,23:1.353418115,41:1.324054342,24:1.341189711,13:1.221057601}

#conditions = [12]
#stags = [0]

#p_model = 'p_vdiv_v'
p_model = 'p_ttcdiff'
#p_model = 'p_eccentricity'
#p_model = 'p_bias_x'

#p_model = 'p_velocity'
#p_model = 'p_ttcs'


# for stag in stags:
#     for con in conditions:
#         dfkh = df[df.condition==con]
#         dfkh = dfkh[dfkh.stagger==stag]
#         model_func = globals()[p_model]
#         best = minimize(m,guess,args=(model_func),bounds=((0.1,10),))        
#         d = {'con':con,'stag':stag,'likelihoods':best.fun, 'minx':best.x[0],'p_model':p_model}
#         print(d)
#         df_d = pd.DataFrame([d])
#         df_res = pd.concat([df_res,df_d],ignore_index=True)


for con in conditions:
    print('condition: ' + str(con))
    dfkh = df[df.condition==con]
    #  dfkh = dfkh[dfkh.stagger==stag]
    model_func = globals()[p_model]
    best = minimize(m,guess,args=(model_func),bounds=((0.1,1),))        
    d = {'con':con,'stag':'all','likelihoods':best.fun, 'minx':best.x[0],'p_model':p_model}
    print(d)
    df_d = pd.DataFrame([d])
    df_res = pd.concat([df_res,df_d],ignore_index=True)





# minimize for each subject separately 
# subjects = df.name.unique()
# df_likelihoods = pd.DataFrame(columns=['name','likelihood'])
# for name in subjects: 
#     dfkh = df[df.name==name]
#     dfkh = dfkh[dfkh.condition.isin(conditions)]
#     dfkh = dfkh[dfkh.stagger==stag]
#     best = minimize(m,guess,args=(p_eccentricity),bounds=((0.1,2),))
#     d={'name':name,'likelihood':best.fun,'best':best.x[0]}
#     df_d = pd.DataFrame([d])
#     df_likelihoods = pd.concat([df_likelihoods,df_d],ignore_index=True)
    

#best = minimize(m,guess,args=(p_eccentricity_times_v),bounds=((None,5),))

#p = selection_likelihood(stdcoeff,x1,v1,x2,v2)
