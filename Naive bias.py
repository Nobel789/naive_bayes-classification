#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:01:51 2023

@author: myyntiimac
"""

#Naibe bayes
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/Users/myyntiimac/Desktop/Social_Network_Ads.csv")
df.head()

#define variable
X = df.iloc[:,[2,3]].values
X
Y = df.iloc[:,-1]

#spliting the variable
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.20,random_state = 0 )


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
ac
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr
#check the bias
bias = classifier.score(X_train, y_train)
bias
#check the variance
variance = classifier.score(X_test, y_test)
variance
#checked  with feature scaling(both, std.scalizer and normalizer)
#Checked without scaling
#bernoli naive bayes dont require feature scaling
#checked all 3 types , Bernoli, gaussian and multinomial
