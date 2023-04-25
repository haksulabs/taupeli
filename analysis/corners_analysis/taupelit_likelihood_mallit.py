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


def p_eccentricity(x,row):
    # calculate variance for ttc-diff based on eccentricity 
    stdcoeff = x 
    x0 = row.x0
    v0 = row.v0
    x1 = row.x1
    v1 = row.v1
    x0endpos = x0 + v0*0.5
    x1endpos = x1 - v1*0.5
 #   print(x0endpos, x1endpos)
    
    sigma1 = abs(x0endpos)*stdcoeff
    sigma2 = abs(x1endpos)*stdcoeff
    
    ttc_diff_var = sigma1**2 + sigma2**2
    
    ttc_diff = np.abs(row.ttcdiff)
    # probability of incorrect answer
    p_wrong = scipy.stats.norm(ttc_diff, np.sqrt(ttc_diff_var)).cdf(0.0)
    if(row.correct == 1):
        p1 = 1-p_wrong
    elif(row.correct == 0):
        p1 = p_wrong 
    else:
        print('hullu')
    return p1

def p_simple(par, row):
    return par

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

df = pd.read_csv('taupelidata_corners_pilotit.csv')

#name = 'kh27_corners'
#name = 'Samuel'
#name = 'corners_kh44'    

conditions = [12]
    
#dfkh = df[df.name==name]
#dfkh = dfkh[dfkh.condition.isin(conditions)]

dfkh = df[df.condition.isin(conditions)]

#print(m(1))
guess = [0.2]
#best = minimize(m,guess,args=(p_simple),bounds=((None,0.9),))
best = minimize(m,guess,args=(p_eccentricity),bounds=((None,0.9),))

print('subject accuracy: ', np.mean(dfkh.correct))
print('optimzed stdcoeff: ', best.x)

#p = selection_likelihood(stdcoeff,x1,v1,x2,v2)
