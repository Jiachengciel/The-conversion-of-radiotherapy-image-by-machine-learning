"""
Created on 2019-05-20
Updated on 2019-06-06
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =================Data Path===========================
PATH = './Images_Test/set_'
PATH_SIMPLE = '/home/nfssrv/liu/Desktop/Machine_Learning/codes/Pytorch_Dose/Images_Test/set_'


# =================Basic Function===========================
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


# =================Data Collection===========================
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


# =================Data Processing===========================
def generate(Path, isNormalize=True):
    """Generate all the input variables -- (7 feature)

    Args:
        Path : the path of the image input and output
        isNormalize: Normalize the input data or not
    Returns:
        X: input of the Neural Network
        Y_: the correct result of the input
    """

    Img_X, _ = get_image(Path=Path, isInput=True)

    if isNormalize:
        Data = normalization(Img_X)
    else:
        Data = Img_X

    # Calculate the dimension of X and Y
    Ncols = Data.shape[1]
    Nrows = Data.shape[0]
    n = Ncols * Nrows
    m = 7
    # minimum value in X like zero padding
    minimum = np.min(Data)

    # Initialize input X
    X = np.zeros((n, m), dtype=float)

    # store the position, intensities
    for i in range(n):
        pos_i = int(i / Ncols + 1)
        pos_j = int(i % Ncols + 1)

        X[i][0] = pos_i  # i
        X[i][1] = pos_j  # j
        X[i][2] = Data[pos_i - 1][pos_j - 1]  # X(i,j)

        if pos_i >= Nrows:
            X[i][3] = minimum
        else:
            X[i][3] = Data[pos_i][pos_j - 1]  # X(i+1,j)

        if (pos_i - 2) < 0:
            X[i][4] = minimum  # i normal
        else:
            X[i][4] = Data[pos_i - 2][pos_j - 1]  # X(i-1,j)

        if pos_j >= Ncols:
            X[i][5] = minimum
        else:
            X[i][5] = Data[pos_i - 1][pos_j]  # X(i,j+1)

        if (pos_j - 2) < 0:
            X[i][6] = minimum
        else:
            X[i][6] = Data[pos_i - 1][pos_j - 2]  # X(i,j-1)

    # normalize i, j
    if isNormalize:
        X[:, 0] = normalization(X[:, 0])  # i normal
        X[:, 1] = normalization(X[:, 1])  # j normal

    # Get the correct result Y_
    Img_Y, _ = get_image(Path=Path, isInput=False)
    # if isNormalize:
    #     Y_ = np.reshape(Img_Y, (n, 1))
    #     Y_ = normalization(Y_)
    # else:
    Y_ = np.reshape(Img_Y, (n, 1))

    return X, Y_


def generate_new(Path, isNormalize=False):
    """Generate all the input variables -- (9 features)

    Args:
        Path : the path of the image input and output
        isNormalize: Normalize the input data or not
    Returns:
        X: input of the Neural Network
        Y_: the correct result of the input
    """

    Img_X, _ = get_image(Path=Path, isInput=True)

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

    # Get the correct result Y_
    Img_Y, _ = get_image(Path=Path, isInput=False)
    if isNormalize:
        Y_ = np.reshape(Img_Y, (n, 1))
        Y_ = normalization(Y_)
    else:
        Y_ = np.reshape(Img_Y, (n, 1))

    return X, Y_


def generate_square(Path, isNormalize=False):
    """Generate all the input variables -- (9 features)

    Args:
        Path : the path of the image input and output
        isNormalize: Normalize the input data or not
    Returns:
        X: input of the Neural Network
        Y_: the correct result of the input
    """

    Img_X, _ = get_image(Path=Path, isInput=True)

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

    # Get the correct result Y_
    Img_Y, _ = get_image(Path=Path, isInput=False)
    if isNormalize:
        Y_ = np.reshape(Img_Y, (n, 1))
        Y_ = normalization(Y_)
    else:
        Y_ = np.reshape(Img_Y, (n, 1))

    return X, Y_


def resample(Data, assement):
    """Resample the data with k-means classify

    Args:
        Data: DataFrame  Y: Data.Y
        assement: k-means classify of data
    Returns:
        Data_resample: Data after resample
    """

    # init the sample seed
    ros = RandomOverSampler(random_state=26)

    # resample
    Data_resample, asse = ros.fit_resample(Data, assement)

    return Data_resample


def generate_resample_data(path):
    """generate data after resample -- for function (generate_new)

    Args:
        path: path of data
    Returns:
        Data_resample: (ndarray) Data after resample
    """

    # get the feutures and target
    X, Y = generate_new(path, isNormalize=False)
    Data = np.c_[X, Y]
    # generate new
    column = ['X(i, j)', 'X(i-1, j-1)', 'X(i-1, j)', 'X(i-1, j+1)', 'X(i, j-1)',
              'X(i, j+1)', 'X(i+1, j-1)', 'X(i+1,j)', 'X(i+1,j+1)',
              'Y']
    df = pd.DataFrame(Data, columns=column)

    # k-means data
    Img_X = get_image(path, isInput=True)[0]
    input_X = Img_X.reshape(Img_X.size, -1)
    kmeans_X = KMeans(n_clusters=2, random_state=2).fit(input_X)
    # assement
    assement = kmeans_X.labels_

    # resample
    Data_resample = resample(df, assement)

    return Data_resample


def generate_resample_data_origin(path):
    """generate data after resample -- for function (generate)

    Args:
        path: path of data
    Returns:
        Data_resample: (ndarray) Data after resample
    """

    # get the feutures and target
    X, Y = generate(path, isNormalize=False)
    Data = np.c_[X, Y]
    # generate
    column = ['i', 'j', 'X(i, j)', 'X(i+1, j)', 'X(i-1, j)', 'X(i, j+1)', 'X(i, j-1)', 'Y']
    df = pd.DataFrame(Data, columns=column)

    # k-means data
    Img_X = get_image(path, isInput=True)[0]
    input_X = Img_X.reshape(Img_X.size, -1)
    kmeans_X = KMeans(n_clusters=2, random_state=2).fit(input_X)
    # assement
    assement = kmeans_X.labels_

    # resample
    Data_resample = resample(df, assement)

    return Data_resample


# ==================Features and Targets Collection for Neural Network============
def get_train_data_origin():
    """get all the train data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [10, 11, 14]

    # first data
    path = PATH_SIMPLE + str(12) + '/'
    Data = generate_resample_data_origin(path)

    for train in TrainExamples:
        path = PATH_SIMPLE + str(train) + '/'
        temp_Data = generate_resample_data_origin(path)
        Data = np.append(Data, temp_Data, axis=0)

    X = Data[:, :7]
    Y = Data[:, 7]

    # Normalization
    ranges = np.max(X, axis=0) - np.min(X, axis=0)
    minValues = X.min(0)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    # Y = normalization(Y)
    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")

    return X, Y, ranges, minValues


def get_train_data():
    """get all the train data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    # TrainExamples = [10, 11, 14, 15, 16]
    TrainExamples = [5, 6, 12, 17, 18, 19, 20, 21, 42, 43, 53, 54, 57, 58, 60]

    # first data
    path = PATH + str(2) + '/'
    Data = generate_resample_data_RF(path)

    for train in TrainExamples:
        path = PATH + str(train) + '/'
        temp_Data = generate_resample_data_RF(path)
        Data = np.append(Data, temp_Data, axis=0)

    X = Data[:, :18]
    Y = Data[:, 18]

    # Normalization
    ranges = np.max(X, axis=0)
    minValues = np.zeros(ranges.shape)
    X = X / np.max(X, axis=0)

    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")

    return X, Y, ranges, minValues


def get_train_data_static():
    """get all the train data of static image from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [5, 6, 7]

    # first data
    path = PATH_SIMPLE + str(1)
    Data = generate_resample_data_origin(path)

    for train in TrainExamples:
        path = PATH_SIMPLE + str(train)
        temp_Data = generate_resample_data_origin(path)
        Data = np.append(Data, temp_Data, axis=0)

    X = Data[:, :7]
    Y = Data[:, 7]

    # Normalization
    ranges = np.max(X, axis=0) - np.min(X, axis=0)
    minValues = X.min(0)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    np.savez('/home/nfssrv/liu/Desktop/Machine_Learning/codes/Pytorch_Dose/model_static/Init.npz', Ranges=ranges,
             MinValues=minValues)
    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")

    return X, Y


# ============================ Random Forest ============================
def generate_resample_data_RF(path):
    """generate data after resample -- for function (generate_new)

    Args:
        path: path of data
    Returns:
        Data_resample: (ndarray) Data after resample
    """

    # get the feutures and target
    X, Y = generate_square(path, isNormalize=False)
    Data = np.c_[X, Y]
    # generate new
    column = ['X(i, j)', 'X(i-1, j-1)', 'X(i-1, j)', 'X(i-1, j+1)', 'X(i, j-1)',
              'X(i, j+1)', 'X(i+1, j-1)', 'X(i+1,j)', 'X(i+1,j+1)',
              'X(i, j)^2', 'X(i-1, j-1)^2', 'X(i-1, j)^2', 'X(i-1, j+1)^2', 'X(i, j-1)^2',
              'X(i, j+1)^2', 'X(i+1, j-1)^2', 'X(i+1,j)^2', 'X(i+1,j+1)^2',
              'Y']
    df = pd.DataFrame(Data, columns=column)

    # k-means data
    Img_X = get_image(path, isInput=True)[0]
    input_X = Img_X.reshape(Img_X.size, -1)
    kmeans_X = KMeans(n_clusters=2, random_state=2).fit(input_X)
    # assement
    assement = kmeans_X.labels_

    # resample
    Data_resample = resample(df, assement)

    return Data_resample


def get_train_data_RF():
    """get all the train data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [3, 6, 7]
    # TrainExamples = [5, 6, 12, 17, 18, 19, 20, 21, 42, 46, 53, 54, 57, 58, 60]

    # first data
    path = PATH + str(1) + '/'
    # Data = generate_resample_data_RF(path)
    X, Y = generate_square(path, isNormalize=False)
    Data = np.c_[X, Y]

    for train in TrainExamples:
        path = PATH + str(train) + '/'
        # temp_Data = generate_resample_data_RF(path)
        X, Y = generate_square(path, isNormalize=False)
        temp_Data = np.c_[X, Y]
        Data = np.append(Data, temp_Data, axis=0)

    X = Data[:, :18]
    Y = Data[:, 18]

    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")

    np.savez('Training_Data_Static_2.npz', X=X, Y=Y)
    return X, Y

# K-means Random Forest
def get_train_data_RF_kmeans(num_cluster=3):
    """get all the train data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [2, 5, 6, 12, 17, 18, 19, 20, 21, 42, 46, 53, 54, 57, 58, 60]
    Data0 = Data1 = Data2 = Data3 = Data4 = np.zeros([1, 19])
    Data_total = [Data0, Data1, Data2, Data3, Data4]

    for train in TrainExamples:
        path = PATH + str(train) + '/'
        print(path)

        X, Y = generate_square(path, isNormalize=False)
        Data = np.c_[X, Y]

        del X, Y

        # K-means
        X_image = get_image(path, isInput=True)[0]
        input_kmeans = X_image.reshape(X_image.size, -1)
        kmeans_x = KMeans(n_clusters=num_cluster, random_state=2).fit(input_kmeans)
        labels = kmeans_x.labels_
        assement = labels.reshape(labels.size, -1)

        # Init the array
        cluster = [i for i in range(num_cluster)]
        mean_image = []
        for j in range(num_cluster):
            index = assement
            value = np.nonzero(index == j)
            mean_temp_image = np.mean(input_kmeans[value[0]])
            mean_image.append(mean_temp_image)

        # Sorted
        Z = zip(mean_image, cluster)
        Z = sorted(Z, reverse=False)
        mean_image_new, cluster_new = zip(*Z)

        # kmeans Data
        for k in range(num_cluster):
            Data_total[k] = np.append(Data_total[k], Data[np.nonzero(assement == cluster_new[k])[0]], axis=0)

    for n in range(num_cluster):
        Data_total[n] = np.delete(Data_total[n], 0, axis=0)
        print(n, " examples: ", Data_total[n].shape[0])
        X = Data_total[n][:, :18]
        Y = Data_total[n][:, 18]
        np.savez('./Train_Data_3/' + str(n) + '.npz', X=X, Y=Y)

    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")


# ====================Normalization Neural Network======================
def generate_resample_data_norm(path):
    """generate data after resample -- for function (generate_new)

    Args:
        path: path of data
    Returns:
        Data_resample: (ndarray) Data after resample
    """

    # get the feutures and target
    X, Y = generate_new(path, isNormalize=True)
    Data = np.c_[X, Y]
    # generate new
    column = ['X(i, j)', 'X(i-1, j-1)', 'X(i-1, j)', 'X(i-1, j+1)', 'X(i, j-1)',
              'X(i, j+1)', 'X(i+1, j-1)', 'X(i+1,j)', 'X(i+1,j+1)', 'Y']
    df = pd.DataFrame(Data, columns=column)

    # k-means data
    Img_X = get_image(path, isInput=True)[0]
    input_X = Img_X.reshape(Img_X.size, -1)
    kmeans_X = KMeans(n_clusters=2, random_state=2).fit(input_X)
    # assement
    assement = kmeans_X.labels_

    # resample
    Data_resample = resample(df, assement)

    return Data_resample


def get_train_data_norm():
    """get all the train data with normalization from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [10, 11, 14, 16]

    # first data
    path = PATH_SIMPLE + str(12) + '/'
    Data = generate_resample_data_norm(path)

    for train in TrainExamples:
        path = PATH_SIMPLE + str(train) + '/'
        temp_Data = generate_resample_data_norm(path)
        Data = np.append(Data, temp_Data, axis=0)

    X = Data[:, :9]
    Y = Data[:, 9]

    # # Normalization
    # ranges = np.max(X, axis=0) - np.min(X, axis=0)
    # minValues = X.min(0)
    # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # np.savez('/home/nfssrv/liu/Desktop/Machine_Learning/codes/Pytorch_Dose/model/Init.npz', Ranges=ranges,
    #          MinValues=minValues)
    print("The training is with these examples: ", TrainExamples)
    print("Finish generating all the train data!")

    return X, Y
