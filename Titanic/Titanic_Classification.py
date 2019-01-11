# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
model=[]
model.append(LogisticRegression())
model.append(KNeighborsClassifier())
model.append(SVC())
model.append(GaussianNB())
model.append(DecisionTreeClassifier())
model.append(RandomForestClassifier())
X=pd.read_csv("train.csv")
y=pd.read_csv("test.csv")
X['Age'].fillna(value=X.Age.mean(),inplace=True)
y['Age'].fillna(value=y.Age.mean(),inplace=True)
X['Embarked'].fillna(method='ffill',inplace=True)
y['Embarked'].fillna(method='ffill',inplace=True)
X_train=X.iloc[:,[0,2,4,5,6,7,9,11]].values
X_test=y.iloc[:,[0,1,3,4,5,6,8,10]].values
y_train=X.iloc[:,1].values
lesex_X_train = LabelEncoder()
leEmbarked_X_train=LabelEncoder()
X_train[:,2] = lesex_X_train.fit_transform(X_train[:,2])
X_train[:,7]=leEmbarked_X_train.fit_transform(X_train[:,7])
onehotencoder_sex = OneHotEncoder(categorical_features = [2])
X_train = onehotencoder_sex.fit_transform(X_train).toarray()
onehotencoder_embarked = OneHotEncoder()
X_train = onehotencoder_embarked.transform(X_train)

