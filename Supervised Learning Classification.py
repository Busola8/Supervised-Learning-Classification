# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:48:15 2024

@author: busola
# """
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("African_crises_dataset.csv")
data
descriptive = data.describe()

unique = data.nunique()

data["banking_crisis"]= data["banking_crisis"].map({"crisis":1,"no_crisis":0})
data["year"]= pd.to_datetime(data["year"])

datahist = data.hist(figsize = (15,10), bins = 10)
# report = ProfileReport(data,title="African Crisis Supervised Learning")
# report.to_file("African Crisis Supervised Learning")

# data= data.drop(["year"],axis = 1)

# data = pd.get_dummies(data, drop_first = True, dtype=int)
# information = data.info()
# corelation = data.corr()

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data.drop(["banking_crisis"],axis = 1))
# scaled_data = pd.DataFrame(scaled_data, columns = scaler.feature_names_in_)

# removed_outliers = scaled_data[(scaled_data >= -3) | (scaled_data <= 3)]
# removed_outliers_data_total = removed_outliers.isnull().sum()

# x = removed_outliers
# y = data["banking_crisis"]


# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 30)

# model = XGBClassifier()
# model.fit(x_train,y_train)
# predict = model.predict(x_test)


# from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
# cm = confusion_matrix(y_test,predict)
# accuracy = accuracy_score(y_test,predict)   #0.97
# recall = recall_score(y_test,predict)   #0.722
# f1score = f1_score(y_test,predict)    #0.81
# classificationreport = classification_report(y_test,predict)
# #    precision    recall  f1-score   support

# #            0       0.97      0.99      0.98       194
# #            1       0.93      0.72      0.81        18

# #     accuracy                           0.97       212
# #    macro avg       0.95      0.86      0.90       212
# # weighted avg       0.97      0.97      0.97       212

