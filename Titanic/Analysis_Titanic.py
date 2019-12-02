#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:46:01 2019

@author: martoso
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, Imputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import collections

df_train = pd.read_csv('train.csv')
df_train.head()

df_train.drop(columns=['Ticket', 'Name', 'Cabin', 'Embarked'], inplace=True)
df_train.head()

def check_missing_data(df):
    columns = df.columns
    for i in columns:
        if pd.isna(df[i]).sum(axis = 0) > 0:
            print('Column: {} has {} NaN value'.format(i, pd.isna(df[i]).sum(axis = 0)))
            
check_missing_data(df_train)

# Fill Values
#df_train['Age'][df_train.Age > 0].mean()
df_train['Age'] = df_train['Age'].fillna(value=df_train['Age'][df_train.Age > 0].mean())
check_missing_data(df_train)

# Forecasters
forecasters_train = df_train.iloc[:, 2:7].values
forecasters_train

# Classes
classes_train = df_train.iloc[:, 1].values
classes_train


#------------------------------------------------------------------------------
df_test = pd.read_csv('test.csv')
df_test.head()


df_test_class = pd.read_csv('gender_submission.csv')
df_test_class.head()


df_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
df_test.head()


#df_test['Age'] = df_test['Age'].fillna(value=df_test['Age'][df_test.Age > 0].mean())
df_test['Age'] = df_test['Age'].fillna(value=df_test['Age'][df_test.Age > 0].mean())
#df_test['Age'][df_test.Age > 0].mean()
df_test['Fare'] = df_test['Fare'].fillna(value=df_test['Fare'][df_test.Age < 500].mean())
check_missing_data(df_test)



forecasters_test = df_test.iloc[:, 1:6].values
forecasters_test


classes_test= df_test_class.iloc[:, 1].values
classes_test

#------------------------------------------------------------------------------
#-----------------------------TREINO-------------------------------------------
# LABEL ENCODER
forecasters_label_encoder = LabelEncoder()
# forecasters_train[:, 0] = forecasters_label_encoder.fit_transform(forecasters_train[:, 0])
forecasters_train[:, 1] = forecasters_label_encoder.fit_transform(forecasters_train[:, 1])
# forecasters_train[:, 3] = forecasters_label_encoder.fit_transform(forecasters_train[:, 3])
# forecasters_train[:, 4] = forecasters_label_encoder.fit_transform(forecasters_train[:, 4])
forecasters_train


# ONE HOT ENCODER
one_hot_encoder = OneHotEncoder(categorical_features=[1])
forecasters_train = one_hot_encoder.fit_transform(forecasters_train).toarray()
forecasters_train


# SCALING
scaler = StandardScaler()
forecasters_train = scaler.fit_transform(forecasters_train)
forecasters_train
#--------------------------TESTE-----------------------------------------------
# LABEL ENCODER
forecasters_test_label_encoder = LabelEncoder()
# forecasters_test[:, 0] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 0])
forecasters_test[:, 1] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 1])
# forecasters_test[:, 3] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 3])
# forecasters_test[:, 4] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 4])
forecasters_test


# ONE HOT ENCODER
one_hot_encoder = OneHotEncoder(categorical_features=[1])
forecasters_test = one_hot_encoder.fit_transform(forecasters_test).toarray()
forecasters_test


# SCALING
scaler = StandardScaler()
forecasters_test = scaler.fit_transform(forecasters_test)
forecasters_test
#---------------------------MODELO---------------------------------------------
estimator = GaussianNB()
estimator.fit(forecasters_train, classes_train)
estimator

predictions = estimator.predict(forecasters_test)
predictions

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------


precision = accuracy_score(classes_test, predictions)
print('Precision: {}%'.format(precision * 100))


matrix = confusion_matrix(classes_test, predictions)
print('Confusion Matrix:'
  '\n\t0\t1\n0:\t{}\t{}'
  '\n1:\t{}\t{}'.format(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))


np.savetxt("predictions.csv", predictions, fmt="%d", deblimiter=",")









