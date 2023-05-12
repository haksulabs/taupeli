# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:24:32 2023

@author: t
"""



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit
from preprocess_taupelidata import corners_preprocess

df = pd.read_csv('taupelidata_corners_pilotit.csv')
df = corners_preprocess(df)

