"""
Created on 2019-06-24
Updated on 2019-06-26
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib

from Pytorch_Input_data import *

# =================Parameters===========================
n_depth = 26  # 26
n_estim = 19
n_features = 11

# =================Store Path===========================
Training = [2, 5, 6, 12, 17, 18, 19, 20, 21, 42, 43, 53, 54, 57, 58, 60]
Training_Static = [1, 3, 5, 6]
PATH_MODEL = './model_RF/RF_kmeans/RandomForest_static_2'

# ================ kmeans training ======================
PARAMETERS = '_depth_' + str(n_depth) + '_estimator_' + str(n_estim) + '_features_' + str(n_features)
PATH_PARAMETERS = PATH_MODEL + PARAMETERS + '.pkl'

# print(PATH_PARAMETERS)
# get the data
X_np, Y_np = get_train_data_RF()
# Split the train and the validaton set for the fitting
X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y_np, test_size=0.2, random_state=26)

accuracy_list = []
iteration_list = []
time_list = []

print("Depth: ", n_depth)
print("Estimators: ", n_estim)
print("Features: ", n_features)
print('\n')

start = time.time()

clf = RandomForestRegressor(n_estimators=n_estim, max_depth=n_depth, max_features=n_features)

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

end = time.time()

r2 = r2_score(Y_test, y_pred) * 100

print("Runing time for features {} is {}s".format(str(n_features), round(end - start, 4)))
# Features
print("Features: " + str(n_features) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
print("Features: " + str(n_features) + " Accuracy: " + str(round(r2, 4)) + "%")
print("\n")

joblib.dump(clf, PATH_PARAMETERS)

# for n_estim in range(1, 20, 1):
#     start = time.time()
#
#     clf = RandomForestRegressor(n_estimators=n_estim, max_depth=n_depth, max_features=n_features)
#
#     clf.fit(X_train, Y_train)
#     y_pred = clf.predict(X_test)
#
#     end = time.time()
#
#     r2 = r2_score(Y_test, y_pred) * 100
#
#
#     print("Runing time for Estimators {} is {}s".format(str(n_estim), round(end-start, 4)))
#     # Features
#     # print("Features: " + str(n_features) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
#     # print("Features: " + str(n_features) + " Accuracy: " + str(round(r2, 4)) + "%")
#     # print("\n")
#     # Depth
#     # print("Depths: " + str(n_depth) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
#     # print("Depths: " + str(n_depth) + " Accuracy: " + str(round(r2, 4)) + "%")
#     # Estimators
#     print("Estimators: " + str(n_estim) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
#     print("Estimators: " + str(n_estim) + " Accuracy: " + str(round(r2, 4)) + "%")
#     print("\n")
#
#     time_list.append(end - start)
#     iteration_list.append(n_estim)
#     accuracy_list.append(r2)


# plt.plot(iteration_list, accuracy_list, color="red")
# plt.xlabel("Estimators")
# plt.ylabel("Accuracy")
# plt.title("Accuracy of Different Estimators")
# # plt.savefig(Path_loss + 'Accuracy_' + PARAMETERS + '.png')
# plt.show()
#
# plt.plot(iteration_list, time_list, color="red")
# plt.xlabel("Estimators")
# plt.ylabel("Time")
# plt.xticks(np.arange(1, 20, 1))
# plt.title("Time of Different Estimators")
# plt.show()
