import pandas as pd

import numpy as np
import os, time , gc

path = os.path.dirname(__file__)
path1 = os.path.join(path, 'dataset/heart.csv')

df = pd.read_csv(path1)

df = df.replace('?', np.nan)

col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = col

#Checking if there is missing values
print("Size={}\nNumber of missing values".format(df.shape))
print(df.isna().sum())

#Replacing missing values with the median
df = df.fillna(df.median())
#Removing duplicates
df = df.drop_duplicates()

#Save preprocessed data
df.to_csv(os.path.join(path, 'dataset/heart.csv'), index=False)
