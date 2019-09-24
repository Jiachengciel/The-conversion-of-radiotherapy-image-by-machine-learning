import numpy as np
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
