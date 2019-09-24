"""
Created on 2019-06-27
Updated on 2019-06-28
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
import os
import time
import numpy as np
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib


from Pytorch_Input_data import *

# Todo
# time.sleep(36000)

# =================Parameters===========================
n_depth = 10  # 26
n_estim = 200
n_features = 5

# =================Store Path===========================
Training = [2, 5, 6, 12, 17, 18, 19, 20, 21, 42, 43, 53, 54, 57, 58, 60]
PATH_ANALYSIS = './Analysis/GBRT/'
PATH_MODEL = './model_GBRT/GBRT'
PARAMETERS = '_depth_' + str(n_depth) + '_estimator_' + str(n_estim) + '_features_' + str(n_features)
PATH_PARAMETERS = PATH_MODEL + PARAMETERS + '.pkl'

print(PATH_PARAMETERS)


# get the data
Data = np.load('Training_Data.npz')
X_np, Y_np = Data['X'], Data['Y']
# Split the train and the validaton set for the fitting
X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y_np, test_size=0.1, random_state=26)

del X_np
del Y_np

print("Depth: ", n_depth)
print("Estimators: ", n_estim)
print("Features: ", n_features)
print('\n')

# print('\nSearch the number of estimators: ')
# param_test1 = {'n_estimators': range(90, 160, 10)}
# gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,
#                                                             min_samples_split=20,
#                                                             subsample=0.8,
#                                                             max_depth=2,
#                                                             random_state=16),
#                         param_grid=param_test1, scoring='r2', iid=False, cv=5)
# gsearch1.fit(X_np, Y_np)
#
# print(gsearch1.cv_results_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
#
#
# print('\nSearch the number of depths and number of sample split: ')
# n_estim = gsearch1.best_params_['n_estimators']
# param_test2 = {'max_depth': range(2, 20, 2), 'min_sample_split': range(10, 170, 20)}
# gsearch2 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,
#                                                             n_estimators=n_estim,
#                                                             subsample=0.8,
#                                                             random_state=16),
#                         param_grid=param_test2, scoring='r2', iid=False, cv=5)
# gsearch2.fit(X_np, Y_np)
#
# print(gsearch2.cv_results_)
# print(gsearch2.best_params_)
# print(gsearch2.best_score_)
#
#
# print('\nSearch the number of max features: ')
# n_depth = gsearch2.best_params_['max_depth']
# n_sample_split = gsearch2.best_params_['min_sample_split']
# param_test3 = {'max_features': range(5, 18, 1)}
# gsearch3 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,
#                                                             n_estimators=n_estim,
#                                                             subsample=0.8,
#                                                             max_depth=n_depth,
#                                                             min_samples_split=n_sample_split,
#                                                             random_state=16),
#                         param_grid=param_test3, scoring='r2', iid=False, cv=5)
# gsearch3.fit(X_np, Y_np)
#
# print(gsearch3.cv_results_)
# print(gsearch3.best_params_)
# print(gsearch3.best_score_)
#
#
# print('\nSearch the number of samples split: ')
# n_features = gsearch3.best_params_['max_features']
# param_test4 = {'subsample': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]}
# gsearch4 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1,
#                                                             max_features=n_features,
#                                                             n_estimators=n_estim,
#                                                             max_depth=n_depth,
#                                                             min_samples_split=n_sample_split,
#                                                             random_state=16),
#                         param_grid=param_test4, scoring='r2', iid=False, cv=5)
# gsearch4.fit(X_np, Y_np)
#
# print(gsearch4.cv_results_)
# print(gsearch4.best_params_)
# print(gsearch4.best_score_)


# for n_estim in range(160, 200, 10):
#     start = time.time()
#
#     clf = GradientBoostingRegressor(n_estimators=n_estim, max_depth=n_depth, max_features=n_features,
#                                     subsample=0.8, random_state=16)
#
#     clf.fit(X_train, Y_train)
#
#     Y_pred = clf.predict(X_train)
#     y_pred = clf.predict(X_test)
#
#     end = time.time()
#
#     r2_train = r2_score(Y_train, Y_pred) * 100
#     r2_test = r2_score(Y_test, y_pred) * 100
#
#
#     print("Runing time for Estimators {} is {}s".format(str(n_estim), round(end-start, 4)))
#
#     # Features
#     # print("Features: " + str(n_features) + " Accuracy Training: " + str(accuracy_score(y_true=Y_train, y_pred=Y_pred)))
#     # print("Features: " + str(n_features) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
#     # print("Features: " + str(n_features) + " Accuracy: " + str(round(r2, 4)) + "%")
#     # print("\n")
#
#     # Depth
#     # print("Depths: " + str(n_depth) + " cost: " + str(mean_squared_error(y_pred, Y_test)))
#     # print("Depths: " + str(n_depth) + " Accuracy: " + str(round(r2, 4)) + "%")
#     # print("\n")
#
#     # Estimators
#     print("Estimators: " + str(n_estim) + " Accuracy Training: " + str(round(r2_train, 4)) + "%")
#     print("Estimators: " + str(n_estim) + " Cost train: " + str(mean_squared_error(Y_pred, Y_train)))
#
#     print("Estimators: " + str(n_estim) + " Accuracy Test: " + str(round(r2_test, 4)) + "%")
#     print("Estimators: " + str(n_estim) + " Cost test: " + str(mean_squared_error(y_pred, Y_test)))
#     print("\n")
#
#     time_list.append(end - start)
#     iteration_list.append(n_estim)
#
#     accuracy_list_1.append(r2_train)
#     accuracy_list_2.append(r2_test)


# for sample_split in range(2, 100, 10):
#
#
#     accuracy_list_1 = []
#     accuracy_list_2 = []
#
#     iteration_list = []
#     time_list = []
#
#     for n_depth in range(2, 20, 2):
#         start = time.time()
#
#         clf = GradientBoostingRegressor(n_estimators=n_estim,
#                                         max_depth=n_depth,
#                                         max_features=n_features,
#                                         min_samples_split=sample_split,
#                                         subsample=0.8, random_state=16)
#
#         clf.fit(X_train, Y_train)
#
#         Y_pred = clf.predict(X_train)
#         y_pred = clf.predict(X_test)
#
#         end = time.time()
#
#         r2_train = r2_score(Y_train, Y_pred) * 100
#         r2_test = r2_score(Y_test, y_pred) * 100
#
#
#         print("Runing time for depths {}, min_sample_split {} is {}s".format(str(n_depth), str(sample_split), round(end-start, 4)))
#
#         # Depth
#         print("Training | " + "Depths: " + str(n_depth) + ",min_sample_split: " + str(sample_split) + "| cost:  " + str(mean_squared_error(Y_pred, Y_train)) + " | Accuracy:" + str(round(r2_train, 4)) + "%")
#         print("Test | " + "Depths: " + str(n_depth) + ",min_sample_split: " + str(sample_split) + "| cost:  " + str(
#             mean_squared_error(y_pred, Y_test)) + " | Accuracy:" + str(round(r2_test, 4)) + "%")
#
#         print("\n")
#
#         time_list.append(end - start)
#         iteration_list.append(n_depth)
#
#         accuracy_list_1.append(r2_train)
#         accuracy_list_2.append(r2_test)
#
#     plt.plot(iteration_list, accuracy_list_1, color="red", label='training accuracy')
#     plt.plot(iteration_list, accuracy_list_2, color="blue", label='test accuracy')
#     plt.legend()
#     plt.xlabel("Depth")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy of Different Depth | sample_split:" + str(sample_split))
#     plt.savefig(PATH_ANALYSIS + '|Depth:2-20|' + "Sample_split=" + str(sample_split) + '.png')
#     plt.show()
#
#     plt.plot(iteration_list, time_list, color="red")
#     plt.xlabel("Depths")
#     plt.ylabel("Time")
#     plt.xticks(np.arange(2, 20, 2))
#     plt.title("Time of Different Estimators" + str(sample_split))
#     plt.show()

print("Start Training...")
start = time.time()

clf = GradientBoostingRegressor(n_estimators=n_estim, max_depth=n_depth, max_features=n_features,
                                subsample=0.8, random_state=16)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)

end = time.time()

r2_train = r2_score(Y_train, Y_pred) * 100
r2_test = r2_score(Y_test, y_pred) * 100

print("Runing time is {}s".format(round(end-start, 4)))

# Depth
print("Training " + "| cost:  " + str(mean_squared_error(Y_pred, Y_train)) + " | Accuracy:" + str(round(r2_train, 4)) + "%")
print("Test " + "| cost:  " + str(mean_squared_error(y_pred, Y_test)) + " | Accuracy:" + str(round(r2_test, 4)) + "%")

# print("Runing time for features {} is {}s".format(str(n_features), round(end-start, 4)))
# # Features
# print("Features: " + str(n_features) + " Accuracy Training: " + str(accuracy_score(y_true=Y_train, y_pred=Y_pred)))
# print("Features: " + str(n_features) + " cost test: " + str(mean_squared_error(y_pred, Y_test)))
# print("Features: " + str(n_features) + " Accuracy Test: " + str(round(r2, 4)) + "%")
# print("\n")

# plt.plot(iteration_list, accuracy_list_1, color="red", label='training accuracy')
# plt.plot(iteration_list, accuracy_list_2, color="blue", label='test accuracy')
# plt.legend()
# plt.xlabel("Estimators")
# plt.ylabel("Accuracy")
# plt.title("Accuracy of Different Estimators")
# # plt.savefig(Path_loss + 'Accuracy_' + PARAMETERS + '.png')
# plt.show()
#
# plt.plot(iteration_list, time_list, color="red")
# plt.xlabel("Estimators")
# plt.ylabel("Time")
# plt.xticks(np.arange(160, 200, 10))
# plt.title("Time of Different Estimators")
# plt.show()

joblib.dump(clf, PATH_PARAMETERS)
