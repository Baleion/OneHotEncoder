# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:17:30 2016

@author: awohl
"""

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

#Get the leaf data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def encode(train, test):
    '''This function encodes the labels for the train and test sets as long as they are dataframes'''
    le = LabelEncoder().fit(train.species)
    ohe = OneHotEncoder()
    labels_le = le.transform(train.species)
    
    classes = list(le.classes_)
    test_ids = test.id
    
    train = train.drop(['species','id'],axis = 1)
    test = test.drop(['id'],axis = 1)
    
    return train, labels, test, test_ids,classes

train, labels, test, test_ids, classes = encode(train,test)

