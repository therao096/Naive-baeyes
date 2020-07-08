# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:53:58 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
salary=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\NAIVE BAYES\\SalaryData_Train.csv")
salary.describe

####function to convert categories to factors
def datatypes(df):
    a=df.dtypes
    g = dict(a)
    le=LabelEncoder()
    for key in g.keys():
        if g[key]==np.object:
           df[key]=le.fit_transform(df[key])
    return df
           
df = datatypes(salary)

colnames=list(df.columns)



ip_columns=colnames[:13]
type(ip_columns)
op_columns=colnames[13]


###training and testing data

Xtrain,Xtest,ytrain,ytest= train_test_split(df[ip_columns],df[op_columns],test_size=0.3, random_state=0)

ignb= GaussianNB()
imnb=MultinomialNB()


###BUILD AND TEST
pred_gnb= ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)


# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten() ##79.37%





confusion_matrix(ytest,pred_mnb) # multinomialNB model
pd.crosstab(ytest.values.flatten(),pred_mnb) # confusion matrix using 
np.mean(pred_mnb==ytest.values.flatten()) #  ##77.32
df['pred']= pred_gnb