# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:53:00 2020

@author: amitdharamsi
"""

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
dataset = pd.read_csv("E:\LetsUpgrade-AI-ML\Assignments\Day 11\general_data.csv")
#Fill null values with 0
dataset.fillna(0, inplace=True)
print(dataset)
#for loop to convert all categorical values to numerical values
for col_name in dataset.columns:
    if(dataset[col_name].dtype == 'object'):
        dataset[col_name]= dataset[col_name].astype('category')
        dataset[col_name] = dataset[col_name].cat.codes
print(dataset)
for col_name1 in dataset.columns:
    stats, p = pearsonr(dataset.Attrition, dataset[col_name1])
    if stats ==0:
        sign=""
    elif stats>0:
        sign = "+ve"
    else:
        sign = "-ve"
    if abs(stats)>=0.7:
        coreln = "Strong "+sign+" corelation"
    elif abs(stats)<0.7 and abs(stats)>0.4:
        coreln = "Moderate "+sign+" corelation"
    elif abs(stats)<=0.4 and abs(stats)>0:
        coreln = "Weak "+sign+" corelation"
    else:
        coreln = "No corelation"
    print("Column =",col_name1,":","r value:",stats,"p value:",p,"--Corelation:",coreln)
    plt.scatter(dataset.Attrition,dataset[col_name1])
    plt.xlabel="Attrition"
    plt.ylabel=col_name1
    plt.show()
print(dataset.corr())
dataset.corr().to_csv("Correlation.csv")



