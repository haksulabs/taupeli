# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:43:54 2023

statistical test for accuracies between conditionts

@author: t
"""
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocess_taupelidata import corners_preprocess

import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('taupelidata_corners_pilotit.csv')
orig_df = corners_preprocess(df)
df = orig_df



model = smf.mixedlm("correct ~ (d_12 + d_23)*stag1", df, groups=df["name"])
result = model.fit()
trial_counts = df['name'].value_counts()
