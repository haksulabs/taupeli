#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:09:12 2022

@author: t
"""

import numpy as np
import scipy.stats

def selection_likelihood(stdcoeff, x1,v1,x2,v2):
    
    sigma1 = abs(x1)*stdcoeff
    sigma2 = abs(x2)*stdcoeff
    
    ttc_diff_var = sigma1**2 + sigma2**2
    
    ttc1 = -x1/v1
    ttc2 = -x2/v2
    
    ttc_diff = ttc1 - ttc2
    
    p1 = scipy.stats.norm(ttc_diff, np.sqrt(ttc_diff_var)).cdf(0.0)
    
    return p1

import matplotlib.pyplot as plt

x1 = -1.0
v1 = 1.0
x2 = 1.0
v2 = -0.9

stdcoeff = 1

p = selection_likelihood(stdcoeff,x1,v1,x2,v2)

print(p)
    
    