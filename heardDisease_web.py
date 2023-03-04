#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# from sklearn.externals import joblib
import joblib
import os

file_data = os.path.join(os.getcwd(), r"C:\Users\SaraAdel\Desktop\heart_disease_web\heart.csv")
heard_disease = pd.read_csv(file_data)

X = heard_disease.drop("target", axis=1)
Y = heard_disease["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0, stratify=Y, train_size=.8 )

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, Y_train)  # apply scaling on training data

Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression', LogisticRegression())])


# In[2]:

def preprocess_new(X_new):
    input_data_as_numpy_array= np.asarray(X_new)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = pipe.predict(input_data_reshaped)
    if (prediction[0]== 0):
        return  '0'
    else:
        return '1'

