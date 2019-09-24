
import numpy as np
from Input_data import *
from  Pytorch_Input_data import *
from kmeans import *
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


TEST_PATH = './Images/set_17/'
path_com = '/home/dcm/data/study_data/patient_1101250/course_2.16.840.1.114337.1527603007/plan_2.16.840.1.114337.1.1.1527603007.0.2/results/control_21/1_epid_dose_converted.epi.content'
path_ref = '/home/dcm/data/study_data/patient_1101250/course_2.16.840.1.114337.1527603007/plan_2.16.840.1.114337.1.1.1527603007.0.2/results/control_21/1_simulation.epi.content'
path_origin = '/home/dcm/data/study_data/patient_1083366/course_2.16.840.1.114337.1522247038/plan_2.16.840.1.114337.1.1.1522247038.0/results/control_6/4_gamma_evaluation.epi.content'

path_img = '/home/nfssrv/liu/Desktop/Machine_Learning/codes/Pytorch_Dose/Images_Test/'


# =================Class of Neural Network==============
# class Neural_Network(nn.Module):
#
#     def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
#         super(Neural_Network, self).__init__()
#
#         self.ANN = nn.Sequential(
#             # 1
#             nn.Linear(input_dim, hidden1_dim),
#             nn.Tanh(),
#             # 2
#             nn.Linear(hidden1_dim, hidden2_dim),
#             nn.Tanh(),
#             # 3
#             nn.Linear(hidden2_dim, hidden3_dim),
#             nn.Sigmoid(),
#         )
#
#         # Linear function for increasing value: 1 --> 1
#         self.out = nn.Linear(hidden3_dim, output_dim)
#
#     def forward(self, X):
#         y = self.ANN(X)
#
#         #  Increasing the value
#         out = self.out(y)
#
#         return out
#
#
# model = Neural_Network(INPUT_NODE, HIDDEN_LAYER1_NODE, HIDDEN_LAYER2_NODE, HIDDEN_LAYER3_NODE, OUTPUT_NODE)
# print("Neural Network structure:\n", model)

# temp = []
# for i in range(52):
#     # get the path of the images, Y_diff, Y_min
#     set_num = 'set_' + str(i + 1)
#     Path_Img = path_img + set_num + '/'
#
#     # get the information of images
#     Y_, _ = get_image(Path_Img, isInput=False)
#     maxValue = np.max(Y_)
#     temp.append(maxValue)


# path_store = './Results/images/'
#
# for i in range(52):
#     set_num = 'set_' + str(i + 1)
#     Path_Img = path_img + set_num + '/'
#
#     Y, _ = get_image(Path_Img, isInput=False)
#
#     plt.figure()
#     plt.title(set_num)
#     plt.imshow(Y)
#
#     Path_store = path_store + set_num
#     plt.savefig(Path_store)



# X, Y_, Y_diff, Y_min = Input_data.generate(TEST_PATH)
# Data = [X, Y_, Y_diff, Y_min]
# # test
# Y_pre, accuracy = Test.test(Data)
#
# Y_pre = np.reshape(Y_pre, (1024, 1024))
# Y_ = np.reshape(Y_, (1024, 1024))
#
# gamma = Test.gamma_index_matrix(Y_, Y_pre, DTA=2.0, DD=0.02)
# fig = plt.figure()
# ax1 = fig.add_subplot(131)
# ax1.set_title('image origin')
# ax1.imshow(Y_)
#
# ax2 = fig.add_subplot(132)
# ax2.set_title('gamma index')
# ax2.imshow(gamma)
#
# ax3 = fig.add_subplot(133)
# ax3.set_title('image predict')
# ax3.imshow(Y_pre)
#
# plt.show()


for i in range(26):
    set_num = 'set_' + str(i+1)
    path = path_img + set_num + '/'
    X = get_image(path, isInput=True)[0]
    Y = get_image(path, isInput=False)[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('image X')
    img1 = ax1.imshow(X)
    colorbar(img1)

    ax2 = fig.add_subplot(122)
    ax2.set_title('image Y')
    img2 = ax2.imshow(Y)
    colorbar(img2)


    plt.tight_layout()
    path_store = path_img + 'Img/' + set_num
    plt.savefig(path_store)



# fig = plt.figure()
# X = get_image(TEST_PATH, isInput=True)[0]
# Y = get_image(TEST_PATH, isInput=False)[0]
#
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
#
# img1 = ax1.imshow(X)
# colorbar(img1)
#
# img2 = ax2.imshow(Y)
# colorbar(img2)
#
# plt.show()
# plt.title("X vs Y set4")
# plt.xlabel("X value")
# plt.ylabel("Y dose")
# ax.scatter(x=X[:, 2], y=Y, color='red', s=1)
# plt.show()
#
# path = PATH_SIMPLE + 'set_6'
# X, Y = generate(path, isNormalize=False)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.tight_layout()
# plt.grid()
#
#
# ax.scatter(x=X[:,0], y=Y, color='red', s=0.05)
# ax.set_title('i vs Y')
# ax.set_xlabel('i')
# ax.set_ylabel('Y')
# plt.show()
#
# axes[0,0].scatter(x=X[:,0], y=Y, color='red', s=1)
# axes[0,0].set_title('i vs Y')
# axes[0,0].set_xlabel('i')
# axes[0,0].set_ylabel('Y')
#
# axes[0,1].scatter(x=np.power(X[:,0],2), y=Y, color='red', s=1)
# axes[0,1].set_title('i^2 vs Y')
# axes[0,1].set_xlabel('i^2')
# axes[0,1].set_ylabel('Y')
#
# axes[0,2].scatter(x=np.log(X[:,0]), y=Y, color='red', s=1)
# axes[0,2].set_title('log(i) vs Y')
# axes[0,2].set_xlabel('log(i)')
# axes[0,2].set_ylabel('Y')
#
# axes[1,0].scatter(x=X[:,1], y=Y, color='red', s=1)
# axes[1,0].set_title('j vs Y')
# axes[1,0].set_xlabel('j')
# axes[1,0].set_ylabel('Y')
#
# axes[1,1].scatter(x=np.power(X[:,1],2), y=Y, color='red', s=1)
# axes[1,1].set_title('j^2 vs Y')
# axes[1,1].set_xlabel('j^2')
# axes[1,1].set_ylabel('Y')
#
# axes[1,2].scatter(x=np.log(X[:,1]), y=Y, color='red', s=1)
# axes[1,2].set_title('log(j) vs Y')
# axes[1,2].set_xlabel('log(j)')
# axes[1,2].set_ylabel('Y')
#
#
# input_kMeans = KmeansData(Y)
# _, assement = kMeans(input_kMeans, k=2)
# # k-means
# index = assement[:, 0]
# value = np.nonzero(index != assement[0, 0])
#
# corr = assement[:, 0]
# X_resample, corr_resx = ADASYN().fit_resample(X, corr)
# Y_resample, corr_resy = ADASYN().fit_resample(Y, corr)
#
# X_resample_1, corr_resx = ros.fit_resample(X, corr)
# Y_resample_1, corr_resy = ros.fit_resample(Y, corr)


