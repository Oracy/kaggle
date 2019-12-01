#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 02:56:35 2019

@author: martoso
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, Imputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import collections

"""
BEGIN
TRAIN DATA
"""
df_train = pd.read_csv('train.csv')
df_train.head()

df_train.drop(columns=['Ticket', 'Name', 'Cabin', 'Embarked'], inplace=True)
df_train.head()

df_train.describe()

def check_missing_data(df):
    columns = df.columns
    for i in columns:
        if pd.isna(df[i]).sum(axis = 0) > 0:
            print('Column: {} has {} NaN value'.format(i, pd.isna(df[i]).sum(axis = 0)))
            
check_missing_data(df_train)

# df.plot.scatter('Age', 'PassengerId')

# Show Values
df_train['Age'] = df_train['Age'].fillna(value=df_train['Age'][df_train.Age > 0].mean())
# df.loc[df.Age.isna()] = df['Age'][df.Age > 0].mean()
check_missing_data(df_train)

#df.groupby('Embarked').Embarked.value_counts()

#df['Embarked'] = df['Embarked'].fillna(value='S')
#check_missing_data(df)

df_train.head()

forecasters = df_train.iloc[:, 2:9].values
forecasters

classes= df_train.iloc[:, 1].values
classes

# LABEL ENCODER
forecasters_label_encoder = LabelEncoder()
forecasters[:, 1] = forecasters_label_encoder.fit_transform(forecasters[:, 1])
#forecasters[:, 2] = forecasters_label_encoder.fit_transform(forecasters[:, 2])
#forecasters[:, 4] = forecasters_label_encoder.fit_transform(forecasters[:, 4])
forecasters[:, 5] = forecasters_label_encoder.fit_transform(forecasters[:, 5])
forecasters

# SCALING
scaler = StandardScaler()
forecasters = scaler.fit_transform(forecasters)
forecasters

# ONE HOT ENCODER
one_hot_encoder = OneHotEncoder(categorical_features=[1])
forecasters = one_hot_encoder.fit_transform(forecasters).toarray()
forecasters

"""
END
TRAIN DATA
"""
#------------------------------------------------------------------------------
"""
BEGIN
TEST DATA
"""
df_test = pd.read_csv('test.csv')
df_test_class = pd.read_csv('gender_submission.csv')

df_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)


df_test.describe()

def check_missing_data(df):
    columns = df.columns
    for i in columns:
        if pd.isna(df[i]).sum(axis = 0) > 0:
            print('Column: {} has {} NaN value'.format(i, pd.isna(df[i]).sum(axis = 0)))
            
check_missing_data(df_test)

# df.plot.scatter('Age', 'PassengerId')

# Show Values
df_test['Age'] = df_test['Age'].fillna(value=df_test['Age'][df_test.Age > 0].mean())
# df.loc[df.Age.isna()] = df['Age'][df.Age > 0].mean()
check_missing_data(df_test)

#df.groupby('Embarked').Embarked.value_counts()

#df['Embarked'] = df['Embarked'].fillna(value='S')
#check_missing_data(df)

df_test.head()

forecasters_test = df_test.values
forecasters_test

classes_test= df_test_class.iloc[:, 1].values
classes_test

# LABEL ENCODER
forecasters_test_label_encoder = LabelEncoder()
forecasters_test[:, 1] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 1])
#forecasters_test[:, 2] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 2])
#forecasters_test[:, 4] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 4])
forecasters_test[:, 4] = forecasters_test_label_encoder.fit_transform(forecasters_test[:, 4])
forecasters_test

# SCALING
scaler = StandardScaler()
forecasters_test = scaler.fit_transform(forecasters_test)
forecasters_test

# ONE HOT ENCODER
one_hot_encoder_test = OneHotEncoder(categorical_features=[1])
forecasters_test = one_hot_encoder_test.fit_transform(forecasters_test).toarray()
forecasters_test
"""
END
TEST DATA
"""
#------------------------------------------------------------------------------
"""
FORECASTERS TRAIN
"""
forecasters_train = forecasters
classes_train = classes

#------------------------------------------------------------------------------


"""
BEGIN
ML TEST DATA
"""


estimator = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
estimator.fit(forecasters_train, classes_train)
predictions = estimator.predict(forecasters_test)

count = collections.Counter(classes_test)
print('Line Base Classifier {:.2f}'.format(count[0]/(count[0]+count[1])))

precision = accuracy_score(classes_test, predictions)
print('Precision: {:.2f}%'.format(precision * 100))

matrix = confusion_matrix(classes_test, predictions)
print('Confusion Matrix:'
  '\n\t0\t1\n0:\t{}\t{}'
  '\n1:\t{}\t{}'.format(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))

"""
END
ML TEST DATA
"""



"""
BEGIN
TEST DATA
"""
df_test = pd.read_csv('test.csv')
df_test.head()

df_test.drop(columns=['Ticket', 'Name', 'Cabin'], inplace=True)
df_test.head()

df_test.describe()

def check_missing_data(df):
    columns = df.columns
    for i in columns:
        if pd.isna(df[i]).sum(axis = 0) > 0:
            print('Column: {} has {} NaN value'.format(i, pd.isna(df[i]).sum(axis = 0)))
            
check_missing_data(df_test)

df_test.plot.scatter('Age', 'PassengerId')

# Show Values
df_test['Age'] = df_test['Age'].fillna(value=df_test['Age'][df_test.Age > 0].mean())
# df.loc[df.Age.isna()] = df['Age'][df.Age > 0].mean()
check_missing_data(df_test)

df_test['Fare'] = df_test['Fare'].fillna(value=df_test['Fare'][df_test.Fare < 500].mean())
check_missing_data(df_test)

df_test[df_test.isna().any(axis=1)]

df_test.groupby('Embarked').Embarked.value_counts()

df_test['Embarked'] = df_test['Embarked'].fillna(value='S')
check_missing_data(df_test)

df_test.head()

df_test.rename(columns={'Pclass': 'Survived', }, inplace=True)


forecasters = df_test.iloc[:, 2:8].values
forecasters

forecasters_label_encoder_test = LabelEncoder()
forecasters[:, 0] = forecasters_label_encoder.fit_transform(forecasters[:, 0])
forecasters[:, 5] = forecasters_label_encoder.fit_transform(forecasters[:, 5])
forecasters

scaler = StandardScaler()
forecasters = scaler.fit_transform(forecasters)
forecasters
"""
END
TEST DATA
"""

"""
BEGIN
ESTIMATOR
"""

estimator = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
estimator.fit(forecasters_train, classes_train)
predictions = estimator.predict(forecasters_test)

count = collections.Counter(classes_test)
count[0]/(count[0]+count[1])

precision = accuracy_score(classes_test, predictions)
print('Precision: {:.2f}%'.format(precision * 100))

matrix = confusion_matrix(classes_test, predictions)
print('Confusion Matrix:'
  '\n\t0\t1\n0:\t{}\t{}'
  '\n1:\t{}\t{}'.format(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))


"""
END
ESTIMATOR
"""

#estimator = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
#estimator.fit(forecasters, classes)