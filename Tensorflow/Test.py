import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import Forward
import Backward
from Input_data import *

TEST_NUM = 52
TEST_NUM_SIMPLE = 15
TEST_PATH = './Images/'
PATH_FIGURE = './Results/Compare/'

# path_test = './Images/set_2/'  # 93.26   mean:0.44  max: 9.80    # 94.67%  0.45  23.07
# path_test = './Images/set_3/'  # 93.14   mean:0.50  max: 10.86
# path_test = './Images/set_4/'
#
# path_test = './Images/set_17/'  # 99.68%   mean:0.17  max: 1.53
# path_test = './Images/set_18/'  # 94.78%   mean:0.31  max: 11.30
#                                   98.27%   mean:0.23  max: 15.57  # k-means: 72.75% 1.45 15.57
# path_test = './Images/set_19/'  # 98.28%   mean:0.16  max: 4.16
# path_test = './Images/set_20/'  # 95.92%   mean:0.24  max: 8.43
# path_test = './Images/set_21/'  # 99.98%   mean:0.08  max: 1.51    99.62%  mean: 0.27
# path_test = './Images/set_22/'  # 99.70    mean:0.16  max: 1.58
# path_test = './Images/set_23/'  # 94.76%   mean:0.33  max: 11.11
# path_test = './Images/set_24/'  # 99.98%   mean:0.10  max: 1.43
# path_test = './Images/set_25/'  # 98.34%   mean:0.16  max: 4.13
# path_test = './Images/set_37/'  # 70.63%   mean:2.07  max: 21.87
#                                   66.58%   mean:1.50  max: 16.21
# path_test = './Images/set_45/'  # 70.63%   mean:2.07  max: 21.87
path_test = './Images_Simple/set_9/'


def test(Data):
    """Test for other image

        Args:
            Data: input of X and Y_ , structure is [X, Y_]

        Returns:
             y_pred: the predict image y of the input X

        Raises:
            Exception: can't find the model of Neural Network
            Exception: the shape of the input and output aren't same
        """

    # We predict two part in different calculate graph
    g1 = tf.Graph()  # low dose

    X = Data[0]
    Y_ = Data[1]
    maxvalue = Data[2]

    # use the model of Neural Network
    with g1.as_default():
        x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
        y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, None, maxvalue, Is_model_high=False)

        ema = tf.train.ExponentialMovingAverage(Backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            # Get the check point
            ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH_TEST)

            # ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH_LINEAR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                '''without k-means'''
                y_pred = sess.run(y, feed_dict={x: X, y_: Y_})
                return y_pred

            else:
                print("No check point file found!")


def Test_kMeans(Data):
    """Test for other image

    Args:
        Data: input of X and Y_ , structure is [X, Y_]

    Returns:
         y_pred: the predict image y of the input X

    Raises:
        Exception: can't find the model of Neural Network
        Exception: the shape of the input and output aren't same
    """

    # We predict two part in different calculate graph
    g1 = tf.Graph()  # low dose
    g2 = tf.Graph()  # high dosess

    X_Low = Data[0]
    Y_Low = Data[1]
    X_High = Data[2]
    Y_High = Data[3]
    maxvalue = Data[4]

    if X_Low.shape[0] != Y_Low.shape[0]:
        raise Exception(
            "Can't apply the backward function for different quantity of input and output in low dose area!")
    if X_High.shape[0] != Y_High.shape[0]:
        raise Exception(
            "Can't apply the backward function for different quantity of input and output in high dose area!")

    '''without k-means'''
    # X = Data[0]
    # Y_ = Data[1]

    # low dose
    # use the model of Neural Network
    with g1.as_default():
        x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
        y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, None, maxvalue, Is_model_high=False)
        # y = Forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(Backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            # sans k_means
            # ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH)

            # Get the check point
            '''with k-means'''
            ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH_low)

            # ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH_TEST)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                '''with k-means'''
                y_low_pred = sess.run(y, feed_dict={x: X_Low, y_: Y_Low})

            else:
                print("No check point file found!")

    # high dose
    with g2.as_default() as g:
        x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
        y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, None, maxvalue, Is_model_high=True)

        ema = tf.train.ExponentialMovingAverage(Backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            # Get the check point
            ckpt = tf.train.get_checkpoint_state(Backward.MODEL_SAVE_PATH_high)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                y_high_pred = sess.run(y, feed_dict={x: X_High, y_: Y_High})

            else:
                print("No check point file found!")

    return y_low_pred, y_high_pred


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


def get_results(path_img, path_store):
    """Get all the result of the test

    Args:
        path_img: the path having all the image stored
        path_store: store the result of comparision of two images

    Returns:
        Accuracy: a text file storing the gamma index of all the test images
        Comparision Image: the predict image, the gamma index immage and the origin image

    Raises:
        IOError: An error occurred if it can't read the file
    """

    path_write = path_store + '/GammaIndex'
    f = open(path_write, 'a')
    # TestExamples = [17, 18, 19]  # [2, 5, 8, 13]  # [20, 21, 37, 49]  17, 18, 19]
    TestExamples = [22, 23, 24]

    for i in range(TEST_NUM - 1):
        # delete the image bad
        if i + 1 not in TestExamples:
            continue

        # get the path of the images, Y_diff, Y_min
        set_num = 'set_' + str(i + 1)
        Path_Img = path_img + set_num + '/'

        # # get the information of images
        # X, Y_ = generate(Path_Img, isNormalize=True)
        # maxvalue = (get_image(Path_Img, isInput=True)[0].max() / 10000)
        # Data = [X, Y_, maxvalue]
        # # test
        # Y_pre = test(Data)

        # k-means test
        X_low, Y_low, X_high, Y_high, Pos = GenerateClusterData(Path_Img)

        # X, Y_ = generate(path_test, isNormalize=True)
        maxvalue = (get_image(Path_Img, isInput=True)[0].max() / 10000)
        Data = [X_low, Y_low, X_high, Y_high, maxvalue]

        y_low_pred, y_high_pred = Test_kMeans(Data)
        Y = get_image(Path_Img, isInput=False)[0]

        Y_copy = Y.copy()
        # get final results
        Y_copy = Y_copy.reshape(Y.size, 1)
        Y_copy[Pos[0]] = y_low_pred
        Y_copy[Pos[1]] = y_high_pred

        # paint the figure
        img_exact, pixel_size = get_image(Path_Img, isInput=False)
        img_pre = np.reshape(Y_copy, img_exact.shape)

        # write the gamma index
        # k-means
        input_kMeans = KmeansData(img_exact)
        _, assement = kMeans(input_kMeans, k=2)

        gamma_map = Gamma_Map(img_pre, img_exact, pixel_size, pixel_size)

        gamma_copy = gamma_map.copy()
        correct_map = gamma_copy[gamma_copy <= 1]
        precision = correct_map.shape[0] / gamma_map.size
        print(set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
              ", mean: ", str(round(gamma_copy.mean(), 2)),
              ", max: ", str(round(gamma_copy.max(), 2)), "\n")
        f.writelines([set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
                      ", mean: ", str(round(gamma_copy.mean(), 2)),
                      ", max: ", str(round(gamma_copy.max(), 2)), "\n"])

        # store the array
        path_array = path_store + '/Array/' + set_num
        np.savez(path_array, Origin=img_exact, Predict=img_pre, Gamma=gamma_copy)

        # k-means
        index = assement[:, 0]
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

        # plot
        value0 = np.nonzero(index == assement[0, 0])
        gamma_map_all[value0[0]] = np.NaN
        gamma_map_all = np.reshape(gamma_map_all, img_pre.shape)

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
        ax4.axis('off')

        ax1.imshow(img_exact)
        ax2.imshow(img_pre)
        ax3.imshow(gamma_copy)
        ax4.imshow(gamma_map_all)

        # store the figure
        Path_Store = path_store + set_num
        plt.savefig(Path_Store)

        print("Finish ", set_num, "!")

    f.close()


def get_results_simple(path_img, path_store):
    """Get all the result of the test

    Args:
        path_img: the path having all the image stored
        path_store: store the result of comparision of two images

    Returns:
        Accuracy: a text file storing the gamma index of all the test images
        Comparision Image: the predict image, the gamma index immage and the origin image

    Raises:
        IOError: An error occurred if it can't read the file
    """

    path_write = path_store + 'GAI'
    f = open(path_write, 'a')
    TestExamples = [8, 9]

    for i in range(TEST_NUM_SIMPLE - 1):
        # delete the image bad
        if i + 1 not in TestExamples:
            continue

        # get the path of the images, Y_diff, Y_min
        set_num = 'set_' + str(i + 1)
        Path_Img = path_img + set_num + '/'

        # # get the information of images
        X, Y_ = generate(Path_Img, isNormalize=True)
        maxvalue = (get_image(Path_Img, isInput=True)[0].max() / 10000)
        Data = [X, Y_, maxvalue]
        # test
        Y_pre = test(Data)
        Y_copy = Y_pre.copy()

        # k-means test
        X_low, Y_low, X_high, Y_high, Pos = GenerateClusterData(Path_Img)

        # X, Y_ = generate(path_test, isNormalize=True)
        # maxvalue = (get_image(Path_Img, isInput=True)[0].max() / 10000)
        # Data = [X_low, Y_low, X_high, Y_high, maxvalue]
        #
        # y_low_pred, y_high_pred = Test_kMeans(Data)
        # Y = get_image(Path_Img, isInput=False)[0]
        #
        # Y_copy = Y.copy()
        # # get final results
        # Y_copy = Y_copy.reshape(Y.size, 1)
        # Y_copy[Pos[0]] = y_low_pred
        # Y_copy[Pos[1]] = y_high_pred
        #
        # paint the figure
        img_exact, pixel_size = get_image(Path_Img, isInput=False)
        img_pre = np.reshape(Y_copy, img_exact.shape)

        # write the gamma index
        # k-means
        input_kMeans = KmeansData(img_exact)
        _, assement = kMeans(input_kMeans, k=2)

        gamma_map = Gamma_Map(img_pre, img_exact, pixel_size, pixel_size)

        gamma_copy = gamma_map.copy()
        correct_map = gamma_copy[gamma_copy <= 1]
        precision = correct_map.shape[0] / gamma_map.size
        print(set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
              ", mean: ", str(round(gamma_copy.mean(), 2)),
              ", max: ", str(round(gamma_copy.max(), 2)), "\n")
        f.writelines([set_num, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
                      ", mean: ", str(round(gamma_copy.mean(), 2)),
                      ", max: ", str(round(gamma_copy.max(), 2)), "\n"])

        # store the array
        path_array = path_store + 'Array/' + set_num
        np.savez(path_array, Origin=img_exact, Predict=img_pre, Gamma=gamma_copy)

        # k-means
        index = assement[:, 0]
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

        # plot
        value0 = np.nonzero(index == assement[0, 0])
        gamma_map_all[value0[0]] = np.NaN
        gamma_map_all = np.reshape(gamma_map_all, img_pre.shape)

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
        img2 = ax2.imshow(img_pre)
        img3 = ax3.imshow(gamma_copy)
        img4 = ax4.imshow(gamma_map_all)

        colorbar(img1)
        colorbar(img2)
        colorbar(img3)
        colorbar(img4)

        # store the figure
        Path_Store = path_store + set_num
        plt.savefig(Path_Store)

        print("Finish ", set_num, "!")

    f.close()


def main():
    # Test Images Simple
    path_img = './Images_Simple/'
    path_store = './Images_Simple/Gamma_test/'

    # path_img = './Images/'
    # path_store = './Images/Gamma_test/'

    get_results_simple(path_img, path_store)

    # Test All The Images
    # path_img = './Images/'
    # path_store = './Results/gamma_index/'
    #
    # get_results(path_img, path_store)


if __name__ == '__main__':
    main()


    # X_low, Y_low, X_high, Y_high, Pos =Results GenerateClusterData(path_test)
    #
    # # X, Y_ = generate(path_test, isNormalize=True)
    # maxvalue = (get_image(path_test, isInput=True)[0].max() / 10000)
    # Data = [X_low, Y_low, X_high, Y_high, maxvalue]
    #
    # y_low_pred, y_high_pred = test(Data)
    # Y = get_image(path_test, isInput=False)[0]
    #
    # Y_copy = Y.copy()
    # # get final results
    # Y_copy = Y_copy.reshape(Y.size, 1)
    # Y_copy[Pos[0]] = y_low_pred
    # Y_copy[Pos[1]] = y_high_pred
    #
    # Y_pred = Y_copy.reshape(Y.shape)



    # X, Y_ = generate(path_test, isNormalize=True)
    # maxvalue = (get_image(path_test, isInput=True)[0].max() / 10000)
    # Data = [X, Y_, maxvalue]
    # Y_pre = test(Data)
    # Y_copy = Y_pre.copy()

    # get the information of images

    # X_low, Y_low, X_high, Y_high, Pos = GenerateClusterData(path_test)
    # maxvalue = (get_image(path_test, isInput=True)[0].max() / 10000)
    # Data = [X_low, Y_low, X_high, Y_high, maxvalue]
    #
    # # test
    # y_low_pred, y_high_pred = Test_kMeans(Data)
    # Y = get_image(path_test, isInput=False)[0]
    #
    # Y_copy = Y.copy()
    # # get final results
    # Y_copy = Y_copy.reshape(Y.size, 1)
    # Y_copy[Pos[0]] = y_low_pred
    # Y_copy[Pos[1]] = y_high_pred
    #
    # # paint the figure
    # img_exact, pixel_size = get_image(path_test, isInput=False)
    # img_pre = np.abs(np.reshape(Y_copy, img_exact.shape))
    #
    # gamma_map = Gamma_Map(img_pre, img_exact, pixel_size, pixel_size)
    #
    # gamma_copy = gamma_map.copy()
    # correct_map = gamma_copy[gamma_copy <= 1]
    # precision = correct_map.shape[0] / gamma_map.size
    # print(path_test, ":  Gamma index: ", str(round(precision * 100, 2)), "%",
    #       ", mean: ", str(round(gamma_copy.mean(), 2)),
    #       ", max: ", str(round(gamma_copy.max(), 2)), "\n")
