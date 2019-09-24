"""
Created on 2019-08-23
Updated on 2019-08-23
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
# ====Basic librairie=======
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =====Pour Neural Network====
import torch
import torch.nn as nn
from torch.autograd import Variable

# =====Pour Random Forest=====
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


# =================Parameters===========================
# =====Pour Neural Network=======
PATH_PARAMETERS_IMRT = './IMRT/checkpoint_lr5e-05_Epoch80_lambda0.0001.pth.tar'
PATH_PARAMETERS_VMAT= './VMAT/checkpoint_InNode9lr5e-05_Epoch80_lambda0.0001_VMAT_.pth.tar'
PATH_PARAMETERS_STATIC = './Static/checkpoint_lr5e-05_Epoch80_lambda0.0001.pth.tar'

# =====Pour Random Forest========
PATH_PARAMETERS_STATIC_RF = './RandomForest/RandomForest_static_1,3,5,6,7_depth_26_estimator_19_features_11.pkl'
PATH_PARAMETERS_EXACT_RF = './RandomForest/RandomForest_depth_26_estimator_19_features_11.pkl'


# ================ Basic function =======================
def normalization(arr):
    """normalize the array

    Args:
        arr: array
    Return:
        (array): normalization of the array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def standarization(arr):
    """standardize the array

    Args:
        arr: array
    Return:
        (array): standard of the array
    """
    return (arr - np.mean(arr)) / np.std(arr)


def colorbar(image):
    """draw the colorbar of an image

    Args:
        image: image array

    Returns:
         color bar
    """
    ax = image.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(image, cax=cax)


def getPixelSize(Path):
    _, pixel_size = read_information(Path)
    return pixel_size


def getMaxvalue(image):
    return np.max(image)


def getMinvalue(image):
    return np.min(image)


# =============== Generate the image ==================
def read_information(Path):
    """Read the information of the image

    Args:
        Path : the path of the image

    Returns:
        ImPred : matrix of the image
        pixel_size : pixel size of the image, [0] is pixel size on x axis, [1] is pixel size on y axis
    """

    file_in = open(Path, "r")
    # define a little-endian int32 type
    dt = np.dtype('<i4')
    # get the number of the pixel of image
    DataInt = np.fromfile(file_in, dtype=dt, count=2)
    Nrows = DataInt[0]
    Ncols = DataInt[1]

    # get the width and height of the image
    Size = np.fromfile(file_in, dtype='<f', count=4)
    width = Size[2] - Size[0]
    height = Size[3] - Size[1]
    pixel_size_x = float(width / Ncols)
    pixel_size_y = float(height / Nrows)
    pixel_size = [pixel_size_x, pixel_size_y]

    # Read all the intensity of the image
    ImData = np.fromfile(file_in, dtype='<f')
    file_in.close()
    # Resize to an image
    ImPred = np.reshape(ImData, (Ncols, Nrows))

    return ImPred, pixel_size


def get_image(Path, isInput=True):
    """ Read the information of the images

    Args:
        Path : the path of the image
        isInput : whether it is the input of Neural Network

    Returns:
        ImPred : matrix of the image
        pixel_size : pixel size of the image, [0] is pixel size on x axis, [1] is pixel size on y axis

    Raises:
        IOError: An error occurred if it can't read the file
    """

    path = []
    try:
        for root, dirs, files in os.walk(Path):
            for file in files:
                path.append(os.path.join(root, file))

        # simulation is Y(Output), acauired is X(Input)
        if path[0].split('/')[-1].split('.')[0].split('_')[-1] == 'simulation':
            path_X = path[1]
            path_Y = path[0]
        else:
            path_X = path[0]
            path_Y = path[1]

        if isInput:
            path = path_X
        else:
            path = path_Y

    except IOError:
        print("Error: Can't find the file!")

    else:
        ImPred, pixel_size = read_information(path)
        return ImPred, pixel_size


# =====Pour Neural Network==========
def generate(Path, isNormalize=False):
    """Generate all the input variables -- (9 features)

    Args:
        Path : the path of the image input and output
        isNormalize: Normalize the input data or not
    Returns:
        X: input of the Neural Network
        Y_: the correct result of the input
    """

    Img_X, _ = read_information(Path=Path)

    # Padding the Img X
    # minimum value in X like zero padding
    minimum = np.min(Img_X)
    Img_X = np.pad(Img_X, ((1, 1), (1, 1)), 'constant', constant_values=(minimum, minimum))

    if isNormalize:
        Data = normalization(Img_X)
    else:
        Data = Img_X

    # Calculate the dimension of X and Y
    Ncols = Data.shape[1] - 2
    Nrows = Data.shape[0] - 2
    n = Ncols * Nrows
    m = 9

    # Initialize input X
    X = np.zeros((n, m), dtype=float)

    # store the position, intensities
    for i in range(n):
        pos_i = int(i / Ncols + 1)
        pos_j = int(i % Ncols + 1)

        X[i][0] = Data[pos_i][pos_j]  # X(i,j)
        X[i][1] = Data[pos_i - 1][pos_j - 1]  # X(i-1,j-1)
        X[i][2] = Data[pos_i - 1][pos_j]  # X(i-1,j)
        X[i][3] = Data[pos_i - 1][pos_j + 1]  # X(i-1,j+1)
        X[i][4] = Data[pos_i][pos_j - 1]  # X(i,j-1)
        X[i][5] = Data[pos_i][pos_j + 1]  # X(i,j+1)
        X[i][6] = Data[pos_i + 1][pos_j - 1]  # X(i+1,j-1)
        X[i][7] = Data[pos_i + 1][pos_j]  # X(i+1,j)
        X[i][8] = Data[pos_i + 1][pos_j + 1]  # X(i+1,j+1)

    return X

# ======Pour Random Forest==============
def generate_RF(Path, isNormalize=False):
    """Generate all the input variables -- (18 features)

    Args:
        Path : the path of the image input and output
        isNormalize: Normalize the input data or not
    Returns:
        X: input of Random Forest
    """

    Img_X, _ = read_information(Path=Path)

    # Padding the Img X
    # minimum value in X like zero padding
    minimum = np.min(Img_X)
    Img_X = np.pad(Img_X, ((1, 1), (1, 1)), 'constant', constant_values=(minimum, minimum))

    if isNormalize:
        Data = normalization(Img_X)
    else:
        Data = Img_X

    # Calculate the dimension of X and Y
    Ncols = Data.shape[1] - 2
    Nrows = Data.shape[0] - 2
    n = Ncols * Nrows
    m = 18

    # Initialize input X
    X = np.zeros((n, m), dtype=float)

    # store the position, intensities
    for i in range(n):
        pos_i = int(i / Ncols + 1)
        pos_j = int(i % Ncols + 1)

        X[i][0] = Data[pos_i][pos_j]  # X(i,j)
        X[i][1] = Data[pos_i - 1][pos_j - 1]  # X(i-1,j-1)
        X[i][2] = Data[pos_i - 1][pos_j]  # X(i-1,j)
        X[i][3] = Data[pos_i - 1][pos_j + 1]  # X(i-1,j+1)
        X[i][4] = Data[pos_i][pos_j - 1]  # X(i,j-1)
        X[i][5] = Data[pos_i][pos_j + 1]  # X(i,j+1)
        X[i][6] = Data[pos_i + 1][pos_j - 1]  # X(i+1,j-1)
        X[i][7] = Data[pos_i + 1][pos_j]  # X(i+1,j)
        X[i][8] = Data[pos_i + 1][pos_j + 1]  # X(i+1,j+1)

        X[i][9] = X[i][0] ** 2  # X(i,j)
        X[i][10] = X[i][1] ** 2  # X(i-1,j-1)
        X[i][11] = X[i][2] ** 2  # X(i-1,j)
        X[i][12] = X[i][3] ** 2  # X(i-1,j+1)
        X[i][13] = X[i][4] ** 2  # X(i,j-1)
        X[i][14] = X[i][5] ** 2  # X(i,j+1)
        X[i][15] = X[i][6] ** 2  # X(i+1,j-1)
        X[i][16] = X[i][7] ** 2  # X(i+1,j)
        X[i][17] = X[i][8] ** 2

    return X


# ===============================Neural Network=====================================================
# ========Parametres Basic==============
INPUT_NODE = 9
HIDDEN_LAYER1_NODE = 30
HIDDEN_LAYER2_NODE = 5
HIDDEN_LAYER3_NODE = 1
OUTPUT_NODE = 1

# =================Class of Neural Network==============
class Neural_Network(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(Neural_Network, self).__init__()

        self.ANN = nn.Sequential(
            # 1
            nn.Linear(input_dim, hidden1_dim),
            nn.Tanh(),
            # 2
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.Tanh(),
            # 3
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.Sigmoid(),
        )

        # Linear function for increasing value: 1 --> 1
        self.out = nn.Linear(hidden3_dim, output_dim)

    def forward(self, X):
        y = self.ANN(X)

        #  Increasing the value
        out = self.out(y)

        return out

# ================== Conversion by Neural Network =======================
def Conversion_ANN(X, isStatic=False, isVMAT=False):
    """Test for other image

        Args:
            X: input of the image
            isStatic: is it the image static
            isVMAT: is it the image VMAT

        Returns:
             prediction: the predict image dosimétrique of the input X

        Raises:
            Exception: can't find the model of Neural Network
        """

    # Tensor Processing
    X = torch.from_numpy(X)

    # Model Basic
    model = Neural_Network(INPUT_NODE, HIDDEN_LAYER1_NODE, HIDDEN_LAYER2_NODE, HIDDEN_LAYER3_NODE, OUTPUT_NODE)

    # Check whether there is a model
    if isStatic:
        PATH_PARAMETERS = PATH_PARAMETERS_STATIC
    elif isVMAT:
        PATH_PARAMETERS = PATH_PARAMETERS_VMAT
    else:
        PATH_PARAMETERS = PATH_PARAMETERS_IMRT

    IsExists = os.path.exists(PATH_PARAMETERS)
    if IsExists:
        print("Model exists, begin test!!!")

        # Get the Parameters of Model
        checkpoint = torch.load(PATH_PARAMETERS)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        print("No model, try to find it!!!")
        return None

    # Predict the Target
    prediction = model(X.float())

    return prediction.detach().numpy()


def get_results_ANN(path, isStatic=False, isVMAT=False):
    """Get all the result of the test

    Args:
        path: path of the image EPID
        isStatic: conversion for image static (true or false)
        isStatic: conversion for image VMAT (true or false)

    Returns:
        Accuracy: a text file storing the gamma index of all the test images
        Comparision Image: the predict image, the gamma index immage and the origin image

    Raises:
        IOError: An error occurred if it can't read the file
    """

    # Basic for Normalization of the test image
    if isStatic:
        Init_Array = np.load('/Static/Init.npz')
    elif isVMAT:
        Init_Array = np.load('./VMAT/Init_InNode9lr5e-05_Epoch80_lambda0.0001_VMAT_.npz')
    else:
        Init_Array = np.load('./IMRT/Init_lr5e-05_Epoch80_lambda0.0001_9_10_11_14_21_22_.npz')

    ranges = Init_Array['Ranges']
    minValues = Init_Array['MinValues']

    X = generate(Path=path, isNormalize=False)
    X = (X - minValues) / ranges

    # Prediction with Model
    Y_pre = Conversion_ANN(X, isStatic, isVMAT)

    return Y_pre



# ================================================ Random Forest =======================================================

# =================Conversion by Random Forest===============
def Conversion_RF(X, isStatic=False):
    """Test for other image

        Args:
            X: input of the image

        Returns:
             y_pred: the predict image y of the input X

        Raises:
            Exception: can't find the model of Neural Network
        """
    # Check whether there is a model
    if isStatic:
        PATH_PARAMETERS = PATH_PARAMETERS_STATIC_RF
    else:
        PATH_PARAMETERS = PATH_PARAMETERS_EXACT_RF

    IsExists = os.path.exists(PATH_PARAMETERS)
    if IsExists:
        print("Model exists, begin test!!!")
        # Get the Parameters of Model
        clf = joblib.load(PATH_PARAMETERS)

    else:
        print("No model, try to find it!!!")
        return None

    # Predict the Target
    prediction = clf.predict(X)

    return prediction


def get_results_RF(path, isStatic=False):
    """Get all the result of the test

        Args:
            isStatic: test for image static or image exact (true or false)

        Returns:
            Accuracy: a text file storing the gamma index of all the test images
            Comparision Image: the predict image, the gamma index immage and the origin image

        Raises:
            IOError: An error occurred if it can't read the file
        """

    # Get the information of images
    X = generate_RF(path, isNormalize=False)

    # Prediction with Model
    Y_pre = Conversion_RF(X, isStatic)

    return Y_pre



# ========================================== Main Function =============================================================
def main(Path_Image, isANN, Static, VMAT):
    """La conversion de l'image EPID en image dosimétrique

        Args:
            Path_Image: the path of the image EPID
            isANN: Use the model of ANN or not (True or False)
            Static: the image is static or not (True or False)
            VMAT: the image is VAMT or not (True or False)

        Returns:
            Image_dose : image en dose

    """

    X = read_information(Path_Image)[0]

    if Static and VMAT:
        print("VMAT can't be Static the same time!")

    if isANN:
        Y = get_results_ANN(Path_Image, isStatic=Static, isVMAT=VMAT)
    else:
        Y = get_results_RF(Path_Image, isStatic=Static)

    Image_dose = np.reshape(Y, X.shape)

    # ===== dessin =====
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Image EPID')
    ax2.set_title('Image en dose')

    ax1.axis('off')
    ax2.axis('off')

    img1 = ax1.imshow(X)
    img2 = ax2.imshow(Image_dose)

    colorbar(img1)
    colorbar(img2)

    plt.tight_layout()

    plt.show()

    return Image_dose


if __name__ == '__main__':
    Path_Image_EPID = '/home/nfssrv/liu/Desktop/Machine_Learning/codes/Pytorch_Dose/Images/set_26/5_acquired.epi.content'
    isANN = True
    Static = False
    VMAT = False

    Image_dose = main(Path_Image_EPID, isANN, Static, VMAT)






