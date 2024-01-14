# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:48:15 2024

@author: deji
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
data = pd.read_csv("African_crises_dataset.csv")
data
descriptive = data.describe()

unique = data.nunique()
# duplicates = data.drop_duplicates()
# correlation = data.corr()


data["banking_crisis"]= data["banking_crisis"].map({"crisis":1,"no_crisis":0})
data["year"]= pd.to_datetime(data["year"])


# report = ProfileReport(data,title="African Crisis Supervised Learning")
# report.to_file("African Crisis Supervised Learning")

data = pd.get_dummies(data, drop_first = True, dtype=int)
information = data.info()

corelation = data.corr()

x = data.drop(["banking_crisis","year"],axis = 1)
y = data["banking_crisis"]
num_one_targets = int(np.sum(y))

#Set a counter for targets that are 0
zero_targets_counter = 0

# # We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# # Declare a variable that will do that:
indices_to_remove = []

# # Count the number of targets that are 0. 
# # Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(y.shape[0]):
     if y[i] == 0:
         zero_targets_counter += 1
         if zero_targets_counter > num_one_targets:
             indices_to_remove.append(i)

# # Create two new variables, one that will contain the inputs, and one that will contain the targets.
# # We delete all indices that we marked "to remove" in the loop above.
unscaled_inputs_equal_priors = np.delete(x, indices_to_remove, axis=0)
targets_equal_priors = np.delete(y, indices_to_remove, axis=0)

from sklearn import preprocessing
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


x_train,x_test,y_train,y_test = train_test_split(shuffled_inputs,shuffled_targets, test_size = 0.2, random_state = 30)

model = RandomForestClassifier(n_estimators= 10)
model.fit(x_train,y_train)
predict = model.predict(x_test)




mse = mean_squared_error(y_test,predict)
rmse = np.sqrt(mse)
r = metrics.r2_score(y_test,predict)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predict)


# x = data[]
# num_one_targets = int(np.sum(targets_all))

# # Set a counter for targets that are 0 (meaning that the customer did not convert)
# zero_targets_counter = 0

# # We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# # Declare a variable that will do that:
# indices_to_remove = []

# # Count the number of targets that are 0. 
# # Once there are as many 0s as 1s, mark entries where the target is 0.
# for i in range(targets_all.shape[0]):
#     if targets_all[i] == 0:
#         zero_targets_counter += 1
#         if zero_targets_counter > num_one_targets:
#             indices_to_remove.append(i)

# # Create two new variables, one that will contain the inputs, and one that will contain the targets.
# # We delete all indices that we marked "to remove" in the loop above.
# unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
# targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)