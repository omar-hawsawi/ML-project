import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
#print (df.info())
#if df.isnull().sum().sum() == 0:
   # print("No missing values in the dataset.")
#else:  
    #print("Missing values found in the dataset.") 
print(df.info())
