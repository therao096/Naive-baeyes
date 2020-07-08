# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:52:42 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
email_data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\NAIVE BAYES\\sms_raw_NB.csv",encoding = "ISO-8859-1")

import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>2:
            w.append(word)
    return (" ".join(w))

email_data.text=email_data.text.apply(cleaning_text)
email_data.shape
email_data = email_data.loc[email_data.text != " ",:]

def split_into_words(i):
    return (i.split(" "))
###3data1 = data1.loc[data1.text != " ",:]

from sklearn.model_selection import train_test_split
email_train,email_test= train_test_split(email_data,test_size=0.3, random_state=0)

emails_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.text)
all_emails_matrix = emails_bow.transform(email_data.text)
all_emails_matrix.shape

##for training
train_emails_matrix=emails_bow.transform(email_train.text)
train_emails_matrix.shape###(3891,7429)


test_emails_matrix = emails_bow.transform(email_test.text)
test_emails_matrix.shape()##(1668,7429)


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,email_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
accuracy_train_m = np.mean(train_pred_m==email_train.type) # 98%

test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 96%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type) #93.6

test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) # 88.36