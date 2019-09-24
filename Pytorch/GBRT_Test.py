"""
Created on 2019-07-11
Updated on 2019-07-11
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from Pytorch_Input_data import *

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib


# =================Parameters===========================
TEST_PATH = './Images/'

path_img = './Images_Test/'
path_store = './Result/Gamma_test_GBRT_Test_2/'

PATH_RF_EXACT = './model_GBRT/GBRT_depth_10_estimator_200_features_5.pkl'
PATH_PARAMETERS_MY_STATIC = './model/checkpoint_lr5e-05_Epoch80_lambda0.0001.pth.tar'

# =================Test for GBRT===============
def test(X, isStatic=False):
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
        PATH_PARAMETERS = PATH_PARAMETERS_MY_STATIC
    else:
        PATH_PARAMETERS = PATH_RF_EXACT

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


# =================Gamma Test Function==============
def computeGamma(comparison_image, comparison_PixelSize,
                 xref, yref, ref_value,
                 xmin, xmax, ymin, ymax,
                 DD_scale, DTA_scale):
    """Compute the gamma index of one pixel

    Args:
        comparision_image : array of the compare image
        comparison_PixelSize : pixel size of the comparision image
        x, y : position of the pixel
        ref_value : the value of the reference image at this position
        xmin, xmax, ymin, ymax :(int) the scale of the ellipse
        DTA_scale, DD_scale : parameters of the DTA and DD

    Returns:
        gamma_value : the gamma index of this position
    """

    pixel_size_x = comparison_PixelSize[0]
    pixel_size_y = comparison_PixelSize[1]

    gamma_store = []

    for y in range(ymin, ymax):
        # squared Y distance shift
        shift_y = (y - yref) * pixel_size_y
        shift_y *= shift_y

        for x in range(xmin, xmax):
            # squared X distance shift
            shift_x = (x - xref) * pixel_size_x
            shift_x *= shift_x

            # squared distance
            dd_square = shift_x + shift_y
            dd_square *= DTA_scale

            # square value difference
            dv_square = comparison_image[y][x] - ref_value
            dv_square *= dv_square * DD_scale

            # generalized gamma function(squared)
            gamma_square = dv_square + dd_square
            gamma_store.append(gamma_square)

    gamma_value = min(gamma_store)

    return np.sqrt(gamma_value)


def Gamma_Map(comparison_image, reference_image, reference_PixelSize, comparison_PixelSize, DTA=2.00, DD=0.02):
    """Calculate the gamma index matrixmyplot

    Args:
        comparison_image : test dataset
        reference_image : reference dataset, what the `sample` dataset is expected to be
        reference_PixelSize : the pixel size of the reference image
        DTA : tolerance of the agreement distance (mm)
        DD : tolerance of the relative difference of dose (%)

    Returns:
        gamma_image : gamma index matrix./Images/set_17'

    Raises:
        Exception: An error occurred if the comparision dimension isn't the same as reference dimension.
    """

    if reference_image.shape != comparison_image.shape:
        raise Exception("Cannot compute for matrices of different sizes.")

    # Result matrix
    gamma_image = np.ndarray(reference_image.shape, dtype=np.float32)

    size_x = reference_image.shape[1]
    size_y = reference_image.shape[0]
    last_idx_x = size_x - 1
    last_idx_y = size_y - 1

    zone_x = round(DTA / reference_PixelSize[0])
    zone_y = round(DTA / reference_PixelSize[1])

    max_value = getMaxvalue(reference_image)

    # Help scaling variables
    DTA_scale = float(1 / DTA ** 2)
    DD_scale = float(1 / (max_value ** 2 * DD ** 2))

    for y in range(size_y):
        ymin = max(0, y - zone_y)
        ymax = min(last_idx_y, y + zone_y)

        for x in range(size_x):
            ref_value = reference_image[y][x]
            xmin = int(max(0, x - zone_x))
            xmax = int(min(last_idx_x, x + zone_x))
            gamma_image[y][x] = computeGamma(comparison_image, comparison_PixelSize,
                                             x, y, ref_value,
                                             xmin, xmax, ymin, ymax,
                                             DD_scale, DTA_scale)

            print("Gamma of (%d, %d): %f" % (y, x, gamma_image[y][x]))
    return gamma_image


# =================Get Results of Test==============
def get_results(isStatic=False):
    """Get all the result of the test

    Args:
        isStatic: test for image static or image exact (true or false)

    Returns:
        Accuracy: a text file storing the gamma index of all the test images
        Comparision Image: the predict image, the gamma index immage and the origin image

    Raises:
        IOError: An error occurred if it can't read the file
    """

    # The images for Test
    # Image Static
    if isStatic:
        # TestExamples = [4, 5, 7]
        TestExamples = [1, 2, 3, 4, 5, 6, 7]
        print("Test for static: ", TestExamples)

    else:
        # TestExamples = [9, 10, 11, 14, 15, 21, 22]
        # TestExamples = [8, 12, 13, 16, 17, 18, 19, 20, 23]

        # TestExamples = [2, 5, 6, 12, 17, 18, 19, 20, 21, 42, 43, 53, 54, 57, 58, 60]
        TestExamples = [i for i in range(8, 24)]
        print("Test for exact: ", TestExamples)

    # For writing the results of Gamma Test
    # Todo
    # Change the path for multi test
    path_write = path_store + 'GAI'
    f = open(path_write, 'a')

    for i in TestExamples:
        # get the path of the image
        set_num = 'set_' + str(i)
        Path_Img = path_img + set_num + '/'

        # Get the information of images
        X, Y_ = generate_square(Path_Img, isNormalize=False)

        # Prediction with Model
        Y_pre = test(X, isStatic)
        Y_copy = Y_pre.copy()

        # Paint the figure
        img_exact, pixel_size = get_image(Path_Img, isInput=False)
        img_exact = np.reshape(Y_, img_exact.shape)
        img_prediction = np.reshape(Y_copy, img_exact.shape)

        # ========== Gamma Test for all the image ==============
        gamma_map = Gamma_Map(img_prediction, img_exact, pixel_size, pixel_size)

        gamma_copy = gamma_map.copy()
        correct_map = gamma_copy[gamma_copy <= 1]
        precision = correct_map.shape[0] / gamma_map.size
        print(set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
              ", mean: ", str(round(gamma_copy.mean(), 2)),
              ", max: ", str(round(gamma_copy.max(), 2)), "\n")
        f.writelines([set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
                      ", mean: ", str(round(gamma_copy.mean(), 2)),
                      ", max: ", str(round(gamma_copy.max(), 2)), "\n"])

        # Store the array
        path_array = path_store + 'Array/' + set_num
        np.savez(path_array, Origin=img_exact, Predict=img_prediction, Gamma=gamma_copy)

        # ================== K-means ==========================
        input_kmeans = img_exact.reshape(img_exact.size, -1)
        kmeans_x = KMeans(n_clusters=2, random_state=2).fit(input_kmeans)
        labels = kmeans_x.labels_
        assement = labels.reshape(labels.size, -1)

        # Gamma Test for K-means image
        index = assement
        value = np.nonzero(index != assement[0, 0])
        gamma_map_all = np.reshape(gamma_map, (gamma_map.size, 1))
        gamma_map_kMeans = gamma_map_all[value[0]]

        gamma_map_kMeans_copy = gamma_map_kMeans.copy()
        correct_map_kMeans = gamma_map_kMeans_copy[gamma_map_kMeans_copy <= 1]
        precision_kMeans = correct_map_kMeans.size / gamma_map_kMeans.size
        print("\t", "Gamma index k-means: ", str(round(precision_kMeans * 100, 2)), "%",
              ", mean: ", str(round(gamma_map_kMeans.mean(), 2)), "\n\n")
        f.writelines(["\t", "Gamma index k-means: ", str(round(precision_kMeans * 100, 2)), "%",
                      ", mean: ", str(round(gamma_map_kMeans.mean(), 2)), "\n\n"])

        # =============== Visualization =======================
        value0 = np.nonzero(index == assement[0, 0])
        gamma_map_all[value0[0]] = np.NaN
        gamma_map_all = np.reshape(gamma_map_all, img_prediction.shape)

        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title('Dose origin')
        ax2.set_title('Dose predict')
        ax3.set_title('Gamma test all')
        ax4.set_title('Gamma test K-means')

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        img1 = ax1.imshow(img_exact)
        img2 = ax2.imshow(img_prediction)
        img3 = ax3.imshow(gamma_copy)
        img4 = ax4.imshow(gamma_map_all)

        colorbar(img1)
        colorbar(img2)
        colorbar(img3)
        colorbar(img4)

        plt.tight_layout()
        # Store the figure
        if isStatic:
            Path_Store = path_store + set_num + PATH_PARAMETERS_MY_STATIC.split('/')[-1] + '.png'
        else:
            Path_Store = path_store + set_num + PATH_RF_EXACT.split('/')[-1].split('.')[0] + '.png'

        plt.savefig(Path_Store)

        print("Finish ", set_num, "!")

    f.close()


# =================Test Begin==============
def main():
    get_results(isStatic=False)


if __name__ == '__main__':
    # time.sleep(38000)
    main()