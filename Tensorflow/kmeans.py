import numpy as np
from Input_data import *
from matplotlib import pyplot as plt


def euclidianDistance(x1, x2):
    """define the euclidian metric

    Args:
        x1, x2: (array) the position of two point

    Returns:
        distance: distance of two point
    """
    return np.sqrt(np.sum(np.power(x1-x2, 2), axis=1))


def initCentroid(data, k):
    """Init the centroids of all the data

    Args:
         data: all the data waiting for clustering
         k: quantity of the centroids

    Returns:
        centroids: (array) the position and the feature of the centroids
    """

    n = data.shape[1]  # dimension of the features
    centroids = np.zeros((k, n))
    for i in range(n):
        # random init the centroids in the range of each features
        minJ = np.min(data[:, i])
        maxJ = np.max(data[:, i])
        rangeJ = float(maxJ - minJ)
        centroids[:, i] = np.transpose(np.mat(minJ + rangeJ * np.random.rand(k, 1)))

    return centroids


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


def kMeans(data, k):
    """define the algorithm K-means

    Args:
        data: all the data
        k: quantity of the centroids

    Returns:
        centroids:(array) the centre of the data
        clusterAssement: (array) the result of clustering of each data
    """

    # random init the centroids of the data
    centroids = initCentroid(data, k)
    m, n = data.shape

    # distribute
    # the first clone tell which cluster this data belongs to
    # the second clone tell the distance between the centroid and the data
    clusterAssement = np.zeros((m, 2))
    clusterChanged = True
    iterCount = 0

    while clusterChanged:
        iterCount += 1
        clusterChanged = False

        dist_matrix = np.zeros((m, 2))  # a matrix storing the distance for comparision
        # calculate the distance between each data and each centroid
        for i in range(k):
            centroids_intensity = centroids[i, 2]
            intensity_all = data[:, 2].reshape(-1, 1)
            eucDist = euclidianDistance(intensity_all, centroids_intensity)
            dist_matrix[:, i] = eucDist

        minDist = np.min(dist_matrix, axis=1)
        minIndex = dist_matrix.argmin(axis=1)

        # update its cluster

        if not (clusterAssement[:, 0] == minIndex).all():
            clusterChanged = True
            clusterAssement[:, 0] = minIndex
            clusterAssement[:, 1] = minDist

        # update the centroids
        index_all = clusterAssement[:, 0]
        for j in range(k):
            # get all the points belongs to cluster
            value = np.nonzero(index_all == j)
            pointInCluster = data[value[0]]
            centroids[j, :] = np.mean(pointInCluster, axis=0)

        print("K-means iteration: %d" % iterCount)

    showCluster(data, 1024, 1024, centroids, clusterAssement)
    return centroids, clusterAssement


def showCluster(data, cols, rows, centroid, clusterAssement):
    """show the result of the K-means

    Args:
        data: all the data
        cols, rows: clones and rows of the origin picture
        centroid: centroid of the cluster
        clusterAssement: the result of clustering of each data

    Returns:
        None
    """

    k = centroid.shape[0]
    Pic = np.zeros([rows, cols])

    index_all = clusterAssement[:, 0]
    for i in range(k):
        value = np.nonzero(index_all == i)
        pointInCluster = data[value[0]]
        quantity = pointInCluster.shape[0]

        for l in range(quantity):
            y = int(pointInCluster[l][0])
            x = int(pointInCluster[l][1])
            Pic[y][x] = i*10

    plt.figure('k-means')
    plt.imshow(Pic)
    plt.show()
