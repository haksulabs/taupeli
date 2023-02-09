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


def selection_likelihood(stdcoeff, x0,v0,x1,v1):
    
    x0endpos = x0 + v0*0.5
    x1endpos = x1 - v1*0.5
 #   print(x0endpos, x1endpos)
    
    sigma1 = abs(x0endpos)*stdcoeff
    sigma2 = abs(x1endpos)*stdcoeff
    
    ttc_diff_var = sigma1**2 + sigma2**2
    
    ttc0 = -x0/v0
    ttc1 = x1/v1
    
    #print('ttc0: ',ttc0)
    #print('ttc1: ',ttc1)
    ttc_diff = ttc0 - ttc1
    #print('ttcdiff: ',  ttc_diff)
    #print('variance: ', ttc_diff_var)
    
    p1 = scipy.stats.norm(ttc_diff, np.sqrt(ttc_diff_var)).cdf(0.0)
   # print('p1 : ', p1)
    
    return p1

def simple(prob):
    return prob

def m(stdcoeff):
  
    #tdcoeff = 0.1
    oikein=[]
    vaarin=[]
    
    likelihoods=[]
    for i, row in dfkh.iterrows():
         
        # print('ttc0: ',ttc0)
        # print('ttc1: ',ttc1)
        # ttc_diff = ttc0 - ttc1
        
     
        
        p = stdcoeff
        
        
        #p = selection_likelihood(stdcoeff,row.x0,row.v0,row.x1,row.v1)
        #if(row.ttcdiff<0):
        #    p = 1-p
            
        
        print(p, row.correct)
        if(row.correct):
            oikein.append(p)
            likelihoods.append(p)
        else:
            vaarin.append(p)
            likelihoods.append(1-p)
        
    print('tama ' , np.mean(oikein + vaarin))
    print('likelihoods summa', np.sum(-np.log(likelihoods)))
    return np.sum(-np.log(likelihoods))

df = pd.read_csv('taupelidata_corners_pilotit.csv')

#name = 'kh27_corners'
#name = 'Samuel'
name = 'corners_kh44'    

conditions = [12,34]
    
dfkh = df[df.name==name]
dfkh = dfkh[dfkh.condition.isin(conditions)]


#print(m(1))
guess = [0.2]
best = minimize(m,guess,bounds=((None,0.9),))


print('subject accuracy: ', np.mean(dfkh.correct))
print('optimzed stdcoeff: ', best.x)

#p = selection_likelihood(stdcoeff,x1,v1,x2,v2)


    
    