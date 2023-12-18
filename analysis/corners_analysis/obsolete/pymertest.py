import os
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lm
from pymer4.models import Lmer

os.environ['R_HOME'] = 'C:/Users/t/anaconda3/envs/pymer4/Lib/R'

# Load and checkout sample data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
print(df.head())

model = Lm("DV ~ IV1 + IV2", data=df)

# Fit it
print(model.fit())



# multi level models
print('--------------------------------------------')
print('multilevel model')

model = Lmer("DV ~ IV2 + (IV2|Group)", data=df)

print(model.fit())