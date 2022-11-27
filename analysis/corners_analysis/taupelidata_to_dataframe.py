# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:00:24 2022

@author: t

taupeli logfiles from directory mypath to dataframe
"""
import pandas as pd

from os import listdir
from os.path import isfile, join
import json

import re

column_names = ['name','type', 'v0', 'v1', 'x0', 'x1','v2','v3','x2','x3', 'ttcdiff', 'minttc', 'disappear', 'correct', 'reply_ttc_time', 'reply_time', 'score', 'n_trials']

df = pd.DataFrame(columns= column_names)

#mypath = '/Users/t/Documents/taupeli_data/noise/'

mypath = 'C:/Users/t/iCloudDrive/Documents/taupeli_data/corners_lab_pilot/'

#mypath = '/Users/t/Documents/taupeli_data/corners_lab_pilot/'

#mypath = '/Users/t/Documents/taupeli_data/two_and_one/'

#mypath = 'c:/users/t/duuni/taupeli_data/pilotit/'
#mypath = 'c:/users/t/duuni/taupeli_data/eri_lahtopiste/'
#mypath = 'c:/users/t/duuni/taupeli_data/sama_lahtopiste/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in onlyfiles:

    foo = mypath+file
    
    p = re.compile('\w+')
    
    name = p.match(file).group()
    
    with open(foo) as f:
        lines = f.readlines()
    
    
    for line in lines:  
        js = json.loads(line)
        js['name'] = name
        df = df.append(js,ignore_index=True)
    
df.to_csv('taupelidata_corners_pilotit.csv')
