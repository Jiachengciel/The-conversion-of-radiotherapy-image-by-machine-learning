import numpy as np
import os
from kmeans import kMeans
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PATH = './Images/set_'
PATH_SIMPLE = './Images_Simple/set_'


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

        if isInput:
            path = path[1]
        else:
            path = path[0]

    except IOError:
        print("Error: Can't find the file!")

    else:
        ImPred, pixel_size = read_information(path)
        return ImPred, pixel_size


def generate(Path, isNormalize=True):
    """Generate all the input variables

    Args:
        Path : the path of the image input and output

    Returns:
        X: input of the Neural Network
        Y_: the correct result of the input
    """

    Img_X, _ = get_image(Path=Path, isInput=True)

    '''without k-means'''
    if isNormalize:
        Data = normalization(Img_X)
    # with k-means
    else:
        Data = Img_X

    # Calculate the dimension of X and Y
    Ncols = Data.shape[1]
    Nrows = Data.shape[0]
    n = Ncols * Nrows
    m = 7
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
            X[i][3] = 0
        else:
            X[i][3] = Data[pos_i][pos_j - 1]  # X(i+1,j)

        if (pos_i - 2) < 0:
            X[i][4] = 0  # i normal

        else:
            X[i][4] = Data[pos_i - 2][pos_j - 1]  # X(i-1,j)

        if pos_j >= Ncols:
            X[i][5] = 0
        else:
            X[i][5] = Data[pos_i - 1][pos_j]  # X(i,j+1)

        if (pos_j - 2) < 0:
            X[i][6] = 0
        else:
            X[i][6] = Data[pos_i - 1][pos_j - 2]  # X(i,j-1)

    # normalize i, j
    if isNormalize:
        X[:, 0] = normalization(X[:, 0])  # i normal
        X[:, 1] = normalization(X[:, 1])  # j normal

    # Get the correct result Y_
    Img_Y, _ = get_image(Path=Path, isInput=False)
    Y_ = np.reshape(Img_Y, (n, 1))

    return X, Y_


def get_train_data():
    """get all the train data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [8, 9, 10, 11, 12, 14]
    # from path set_22 to set_35
    path = PATH_SIMPLE + str(5) + '/'
    X, Y_ = generate(path, isNormalize=True)
    maxvalue = (get_image(path, isInput=True)[0].max() / 10000)

    for train in TrainExamples:
        path = PATH + str(train) + '/'
        temp_X, temp_Y = generate(path, isNormalize=True)
        X = np.append(X, temp_X, axis=0)
        Y_ = np.append(Y_, temp_Y, axis=0)

    print("Finish generating all the train data!")

    return X, Y_, maxvalue


def get_test_data():
    """get all the test data from some paths

    Returns:
        X: Input data
        Y_: Compare data
    """

    TrainExamples = [13]
    path = PATH_SIMPLE + str(4) + '/'
    X, Y_ = generate(path, isNormalize=True)
    # maxvalue = (get_image(path, isInput=True)[0].max() / 10000)

    for train in TrainExamples:
        path = PATH + str(train) + '/'
        temp_X, temp_Y = generate(path, isNormalize=True)
        X = np.append(X, temp_X, axis=0)
        Y_ = np.append(Y_, temp_Y, axis=0)

    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    Y_ = Y_[permutation]
    print("Finish generating all the test data!")

    return X, Y_


def get_data_kMeans():
    """get all the data after k-means

    """
    path = PATH + str(21) + '/'
    X_low, Y_low, X_high, Y_high, Pos = GenerateClusterData(path)

    # for i in range(22, 27):
    #     path = PATH + str(i) + '/'
    #     temp_X_low, temp_Y_low, temp_X_high, temp_Y_high, _ = GenerateClusterData(path)
    #     X_low = np.append(X_low, temp_X_low, axis=0)
    #     X_high = np.append(X_high, temp_X_high, axis=0)
    #
    #     Y_low = np.append(Y_low, temp_Y_low, axis=0)
    #     Y_high = np.append(Y_high, temp_Y_high, axis=0)

    permutation = np.random.permutation(X_high.shape[0])
    X = X_high[permutation]
    Y_ = Y_high[permutation]
    print("Finish generating all the data!")

    return X, Y_


def KmeansData(Img):
    """generate the data for K-means

    Args:
        Img: array of image

    Returns:
        Input: input for K-means
    """

    Ncols = Img.shape[1]
    Nrows = Img.shape[0]
    n = Ncols * Nrows
    m = 3
    # Initialize input
    Input = np.zeros((n, m), dtype=float)

    for i in range(n):
        pos_i = int(i / Ncols)
        pos_j = int(i % Ncols)

        Input[i][0] = pos_i  # i
        Input[i][1] = pos_j  # j
        Input[i][2] = Img[pos_i][pos_j]  # intensity(i, j)

    return Input


def GenerateClusterData(Path, k=2):
    """divide all the data into two cluster: high dose and low dose

    Args:
        Path: path of the file which consist the input and output image
        k: number of the cluster, here we define two cluster

    Returns:
        X_high, Y_high: high dose area of input X and output Y_
        X_low, Y_low: low dose area of inpute X and ouput Y_
        Pos[Pos_x_low, Pos_y_low, Pos_x_high, Pos_y_high]: store the position of data in low area and high area
    """

    # get origin data
    X, Y_ = generate(Path, isNormalize=True)
    # get data after K-means
    X_origin, _ = get_image(Path, isInput=True)
    kmeans_X = KmeansData(X_origin)
    _, assement = kMeans(kmeans_X, k)

    # get all the points belongs to cluster
    index_all = assement[:, 0]

    for i in range(k):
        value = np.nonzero(index_all == i)
        # we use the first value for defining the low dose area
        if assement[0, 0] == i:
            Pos_low = value[0]
            X_low = X[Pos_low]
            Y_low = Y_[Pos_low]
        else:
            Pos_high = value[0]
            X_high = X[Pos_high]
            Y_high = Y_[Pos_high]

    # store the position of each data for further integration
    Pos = [Pos_low, Pos_high]

    # normalization
    # for i in range(2, X.shape[1]):
    #     X_low[:, i] = normalization(X_low[:, i])
    #     X_high[:, i] = normalization(X_high[:, i])

    # Y_low = normalization(Y_low)
    # Y_high = normalization(Y_high)

    return X_low, Y_low, X_high, Y_high, Pos


def colorbar(map):
    ax = map.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(map, cax=cax)


def getPixelSize(Path):
    _, pixel_size = read_information(Path)
    return pixel_size


def getMaxvalue(image):
    return np.max(image)


def getMinvalue(image):
    return np.min(image)
