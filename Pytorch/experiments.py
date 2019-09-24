import numpy as np
from Input_data import *
from Pytorch_Input_data import *
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =================Neural Network=======================
INPUT_NODE = 9
HIDDEN_LAYER1_NODE = 30
HIDDEN_LAYER2_NODE = 5
HIDDEN_LAYER3_NODE = 1
HIDDEN_LAYER4_NODE = 1
OUTPUT_NODE = 1

TEST_PATH = './Images/'
path_store = './Result/Analysis_RF/Test/'
path_img = './Images/'

#
# for i in range(68):
#     set_num = 'set_' + str(i+1)
#     path = path_img + set_num + '/'
#     X = get_image(path, isInput=True)[0]
#     Y = get_image(path, isInput=False)[0]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(121)
#     ax1.set_title('image X')
#     img1 = ax1.imshow(X)
#     colorbar(img1)
#
#     ax2 = fig.add_subplot(122)
#     ax2.set_title('image Y')
#     img2 = ax2.imshow(Y)
#     colorbar(img2)
#
#
#     plt.tight_layout()
#     path_store = path_img + 'Img/' + set_num
#     plt.savefig(path_store)

# get_train_data_RF()

# Todo
# Analysis
# set_num = 'set_' + str(58)
# path = path_img + set_num + '/'
# Y = get_image(path, isInput=False)[0]
# Y_pixel = np.reshape(Y, (Y.size, 1))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(Y_pixel, 20)
# ax.set_title(set_num)
# plt.xlabel('Value')
# plt.ylabel('Nombre')
# plt.show()


# =========================== Analysis ============================================
# TestExamples = [2,5]
# TestExamples = [2, 5, 6, 12, 17, 18, 19, 20, 21, 42, 43, 53, 54, 57, 58, 60]
# TestExamples = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# path_write = path_store + "Analysis_GAI"
# f = open(path_write, 'a')
#
# for i in TestExamples:
#     set_num = 'set_' + str(i)
#     print(set_num)
#     f.writelines([set_num, "\n"])
#
#     # ================== Parameters =========================
#     Array = np.load('./Result/Gamma_test_RF_3/Array/Test/' + set_num + '.npz')
#     img_exact = Array['Origin']
#     img_prediction = Array['Predict']
#     gamma_map = Array['Gamma']
#     num_cluster = 5
#
#     # ================== K-means ==========================
#     input_kmeans = img_exact.reshape(img_exact.size, -1)
#     kmeans_x = KMeans(n_clusters=num_cluster, random_state=2).fit(input_kmeans)
#     labels = kmeans_x.labels_
#     assement = labels.reshape(labels.size, -1)
#
#     # =============== Show the kmeans map =========================
#     img_kmeans = np.reshape(assement, (1024, 1024))
#     fig = plt.figure()
#     plt.title(set_num)
#     img = plt.imshow(img_kmeans)
#     colorbar(img)
#     plt.tight_layout()
#     plt.savefig(path_store + set_num + ".png")
#
#     n_bord = 0
#     # # Gamma Test for K-means image
#     for j in range(num_cluster):
#         index = assement
#         value = np.nonzero(index == j)
#         gamma_map_all = np.reshape(gamma_map, (gamma_map.size, 1))
#         gamma_map_kMeans = gamma_map_all[value[0]]
#
#         gamma_map_kMeans_copy = gamma_map_kMeans.copy()
#         correct_map_kMeans = gamma_map_kMeans_copy[gamma_map_kMeans_copy <= 1]
#         precision_kMeans = correct_map_kMeans.size / gamma_map_kMeans.size
#
#         if j == assement[0, 0]:
#             n_bord = value[0].size
#             print("Low  | ", "Index ", str(j),
#                   " | Number of pixel: ", str(value[0].size),
#                   " | Percentage of all: ", str(round(value[0].size / gamma_map.size * 100, 2)), "%",
#                   " | Gamma Test k-means: ", str(round(precision_kMeans * 100, 2)), "%",
#                   " | Mean: ", str(round(gamma_map_kMeans.mean(), 2)))
#             f.writelines(["Low  | ", "Index ", str(j),
#                           " | Number of pixel: ", str(value[0].size),
#                           " | Percentage: ", str(round(value[0].size / gamma_map.size * 100, 2)), "%",
#                           " | Gamma Test k-means: ", str(round(precision_kMeans * 100, 2)), "%",
#                           " | Mean: ", str(round(gamma_map_kMeans.mean(), 2)), "\n"])
#
#         else:
#             print("High | ", "Index ", str(j),
#                   " | Number of pixel: ", str(value[0].size),
#                   " | Percentage of all: ", str(round(value[0].size / gamma_map.size * 100, 2)), "%",
#                   " | Percentage of Centre: ", str(round(value[0].size / (gamma_map.size - n_bord) * 100, 2)), "%",
#                   " | Gamma Test k-means: ", str(round(precision_kMeans * 100, 2)), "%",
#                   " | Mean: ", str(round(gamma_map_kMeans.mean(), 2)))
#             f.writelines(["High | ", "Index ", str(j),
#                           " | Number of pixel: ", str(value[0].size),
#                           " | Percentage: ", str(round(value[0].size / gamma_map.size * 100, 2)), "%",
#                           " | Percentage Centre: ", str(round(value[0].size / (gamma_map.size - n_bord) * 100, 2)), "%",
#                           " | Gamma Test k-means: ", str(round(precision_kMeans * 100, 2)), "%",
#                           " | Mean: ", str(round(gamma_map_kMeans.mean(), 2)), "\n"])
#
#     print("\n")
#     f.writelines("\n")
#
# f.close()
#
#

# set_num = 'set_' + str(17)
# print(set_num)
#
# # ================== Parameters =========================
# Array = np.load('./Result/Gamma_test_RF_3/Array/Train/' + set_num + '.npz')
# img_exact = Array['Origin']
# img_prediction = Array['Predict']
# gamma_map = Array['Gamma']
# num_cluster = 3
#
# # ================== K-means ==========================
# input_kmeans = img_exact.reshape(img_exact.size, -1)
# kmeans_x = KMeans(n_clusters=num_cluster, random_state=2).fit(input_kmeans)
# labels = kmeans_x.labels_
# assement = labels.reshape(labels.size, -1)
#
# # =============== Show the kmeans map =========================
# img_kmeans = np.reshape(assement, (1024, 1024))
# fig = plt.figure()
# plt.title(set_num)
# img = plt.imshow(img_kmeans)
# colorbar(img)
# plt.tight_layout()
# plt.show()

get_train_data_RF_kmeans(num_cluster=2)