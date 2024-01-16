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


# report = ProfileReport(data,title="African Crisis Supervised Learning")
# report.to_file("African Crisis Supervised Learning")
data= data.drop(["year"],axis = 1)
data = pd.get_dummies(data, drop_first = True, dtype=int)
information = data.info()

corelation = data.corr()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(["banking_crisis"],axis = 1))
scaled_data = pd.DataFrame(scaled_data, columns = scaler.feature_names_in_)

removed_outliers = scaled_data[(scaled_data >= -3) | (scaled_data <= 3)]
removed_outliers_data_total = removed_outliers.isnull().sum()

x = removed_outliers
y = data["banking_crisis"]

# num_one_targets = int(np.sum(y))

# #Set a counter for targets that are 0
# zero_targets_counter = 0

# # # We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# # # Declare a variable that will do that:
# indices_to_remove = []

# # # Count the number of targets that are 0. 
# # # Once there are as many 0s as 1s, mark entries where the target is 0.
# for i in range(y.shape[0]):
#       if y[i] == 0:
#           zero_targets_counter += 1
#           if zero_targets_counter > num_one_targets:
#               indices_to_remove.append(i)

# # # Create two new variables, one that will contain the inputs, and one that will contain the targets.
# # # We delete all indices that we marked "to remove" in the loop above.
# unscaled_inputs_equal_priors = np.delete(x, indices_to_remove, axis=0)
# targets_equal_priors = np.delete(y, indices_to_remove, axis=0)

# from sklearn import preprocessing
# scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# shuffled_indices = np.arange(scaled_inputs.shape[0])
# np.random.shuffle(shuffled_indices)

# # Use the shuffled indices to shuffle the inputs and targets.
# shuffled_inputs = scaled_inputs[shuffled_indices]
# shuffled_targets = targets_equal_priors[shuffled_indices]


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 30)

model = XGBClassifier()
model.fit(x_train,y_train)
predict = model.predict(x_test)


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
cm = confusion_matrix(y_test,predict)
accuracy = accuracy_score(y_test,predict)
recall = recall_score(y_test,predict)
f1score = f1_score(y_test,predict)
classificationreport = classification_report(y_test,predict)
