import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import Forward
from Input_data import *


BATCH_SIZE_LOW = 64
BATCH_SIZE_HIGH = 8
BATCH_SIZE = 128

learning_rate_low = 0.0005
learning_rate_high = 0.0004
learning_rate = 0.0003
LEARNING_RATE_BASE = 0.0025
LEARNING_RATE_DECAY = 0.5
REGULARIZER = 0.0002
Epoch = 100
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_SAVE_PATH_low = "./model_low/"
MODEL_SAVE_PATH_high = "./model_high/"
MODEL_SAVE_PATH_TEST = "./model_Test/"
MODEL_SAVE_PATH_ADAM = "./model_Adam/"
MODEL_SAVE_PATH_LINEAR = "./model_Linear/"

MODEL_NAME = "Dose_model"

TEST_PATH = './Images/set_47/'
path_figure = './Results/loss/'


def backward_Kmeans(Data):
    """Bachward for neural network

    Args:
        Data: input data consists low and high input and output, for example:[X_low, Y_low, X_high, Y_high]
        HighDose: to judge whether the input is in the high dose area

    Returns:
        loss_total: loss value for every steps
        check point: store the model of the neural network

    Raises:
         Exception: the shapes of input and output aren't same
    """

    '''comment it when there is not k-means'''
    # We predict two part in different calculate graph
    g1 = tf.Graph()  # low dose
    g2 = tf.Graph()  # high dosess

    X_High = Data[0]
    Y_High = Data[1]
    X_Low = Data[2]
    Y_Low = Data[3]

    if X_Low.shape[0] != Y_Low.shape[0]:
        raise Exception("Can't apply the backward function for different quantity of input and output in low dose area!")
    if X_High.shape[0] != Y_High.shape[0]:
        raise Exception("Can't apply the backward function for different quantity of input and output in high dose area!")

    # At first, we will predict low dose
    Loss_low = []
    with g1.as_default():
        print("This is the process of Low Dose!")
        print("There are %d data in this process." % X_Low.shape[0])
        # Init all the parameters
        global_step = tf.Variable(0, trainable=False)

        STEPS = int(Epoch * X_Low.shape[0] / BATCH_SIZE_LOW) + 1
        epoch = 0

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
            y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, REGULARIZER)

        # lost function
        with tf.name_scope('loss'):
            loss_mse = tf.reduce_mean(tf.square(y - y_))
            loss = loss_mse + tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('loss', loss)

        # Todo
        # LM algorithm

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99).minimize(loss, global_step)
            # train_step = tf.train.MomentumOptimizer(learning_rate_low, momentum=0.9).minimize(loss, global_step)

        # EMA algorithm
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        # ready for storing the model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Get the check point
            # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)

            # Graph
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./logs_low/", sess.graph)

            # Training
            for i in range(STEPS):
                start = (i * BATCH_SIZE_LOW) % int(X_Low.shape[0])
                end = start + BATCH_SIZE_LOW
                # if finish all the data
                if end > X_Low.shape[0]:
                    end = X_Low.shape[0]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: X_Low[start:end], y_: Y_Low[start:end]})

                if i % 5000 == 0:
                    print("Steps are: %d , loss is: %g." % (step, loss_value))
                    rs = sess.run(merged, feed_dict={x: X_Low[start:end], y_: Y_Low[start:end]})
                    writer.add_summary(rs, i)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_low, MODEL_NAME), global_step)

                # a round
                if end == X_Low.shape[0]:
                    # get the results
                    epoch += 1
                    loss_total = sess.run(loss, feed_dict={x: X_Low, y_: Y_Low})
                    Loss_low.append(loss_total)
                    print("After %d epoch(s), loss total is: %g.\n" % (epoch, loss_total))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_low, MODEL_NAME), global_step)

    # High dose

    Loss_high = []
    with g2.as_default():
        print("This is the process of High Dose!")
        print("There are %d data in this process." % X_High.shape[0])
        # Init all the parameters
        global_step = tf.Variable(0, trainable=False)

        STEPS = int(Epoch * X_High.shape[0] / BATCH_SIZE_HIGH) + 1
        epoch = 0

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
            y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, REGULARIZER)

        # lost function
        with tf.name_scope('loss'):
            loss_mse = tf.reduce_mean(tf.square(y - y_))
            loss = loss_mse + tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('loss', loss)

        # Todo
        # LM algorithm

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99).minimize(loss, global_step)
            # train_step = tf.train.MomentumOptimizer(learning_rate_high, momentum=0.9).minimize(loss, global_step)

        # EMA algorithm
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        # ready for storing the model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Graph
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./logs_high/", sess.graph)

            # Training
            for i in range(STEPS):
                start = (i * BATCH_SIZE_HIGH) % int(X_High.shape[0])
                end = start + BATCH_SIZE_HIGH
                # if finish all the data
                if end > X_High.shape[0]:
                    end = X_High.shape[0]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: X_High[start:end], y_: Y_High[start:end]})

                if i % 5000 == 0:
                    print("Steps are: %d , loss is: %g." % (step, loss_value))
                    rs = sess.run(merged, feed_dict={x: X_High[start:end], y_: Y_High[start:end]})
                    writer.add_summary(rs, i)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_high, MODEL_NAME), global_step)

                # a round
                if end == X_High.shape[0]:
                    # get the results
                    epoch += 1
                    loss_total = sess.run(loss, feed_dict={x: X_High, y_: Y_High})
                    Loss_high.append(loss_total)
                    print("After %d epoch(s), loss total is: %g.\n" % (epoch, loss_total))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_high, MODEL_NAME), global_step)

    return Loss_low, Loss_high


def backward(Data):
    """Bachward for neural network

       Args:
           Data: input data consists low and high input and output, for example:[X, Y_]
       Returns:
           loss_total: loss value for every steps
           check point: store the model of the neural network

       Raises:
           Exception: the quantity of input and output are not same
       """

    # data
    graph = tf.Graph()
    X_train = Data[0]
    Y_train = Data[1]
    X_val = Data[2]
    Y_val = Data[3]
    maxvalue = Data[4]

    if X_train.shape[0] != Y_train.shape[0]:
        raise Exception("The quantity of Input X and Compare Y_ are not same!")

    Loss = []
    Loss_val = []

    with graph.as_default():
        print("This is the process of all the Dose!")
        print("There are %d data in training." % X_train.shape[0])
        print("There are %d data in cross validation." % X_val.shape[0])
        print("Features of X: %d" % X_train.shape[1])
        print("Learning rate is: %f" % learning_rate)
        # Init all the parameters
        global_step = tf.Variable(0, trainable=False)

        # multi threads
        # queue = tf.FIFOQueue(capacity=64, dtypes=[tf.float32, tf.float32], shapes=[[7], []])
        # enqueue_op = queue.enqueue_many([X, Y_])
        # X

        STEPS = int(Epoch * X_train.shape[0] / BATCH_SIZE) + 1
        epoch = 0

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
            y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward(x, REGULARIZER, maxvalue, Is_model_high=False)

        # test
        # maxValue = tf.constant(maxValue, dtype=tf.float32)
        # minValue = tf.constant(minValue, dtype=tf.float32)
        # y = y * (maxValue - minValue) + minValue

        # lost function
        with tf.name_scope('loss'):
            loss_mse = tf.reduce_mean(tf.square(y - y_))
            loss = loss_mse + tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('loss', loss)

        # Todo
        # LM algorithm

        # learning_rate = tf.train.exponential_decay(
        #     LEARNING_RATE_BASE,
        #     global_step,
        #     X.shape[0] / BATCH_SIZE,
        #     LEARNING_RATE_DECAY,
        #     staircase=True
        # )

        with tf.name_scope('train'):
            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
            # train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step)
            train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99).minimize(loss, global_step)

        # EMA algorithm
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        # ready for storing the model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Get the check point
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH_TEST)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # begin multi threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            print("Begin the multi threads!")

            # Graph
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./logs/", sess.graph)

            # Training
            for i in range(STEPS):
                start = (i * BATCH_SIZE) % int(X_train.shape[0])
                end = start + BATCH_SIZE
                # if finish all the data
                if end >= X_train.shape[0]:
                    end = X_train.shape[0]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: X_train[start:end], y_: Y_train[start:end]})

                if i % 4000 == 0:
                    print("Steps are: %d , loss is: %g." % (step, loss_value))
                    rs = sess.run(merged, feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
                    writer.add_summary(rs, i)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_TEST, MODEL_NAME), global_step)

                # a round
                if end == X_train.shape[0]:
                    # get the results
                    epoch += 1

                    loss_total = sess.run(loss, feed_dict={x: X_train, y_: Y_train})
                    loss_val = sess.run(loss, feed_dict={x: X_val, y_: Y_val})

                    Loss.append(loss_total)
                    Loss_val.append(loss_val)
                    print("After %d  epoch(s), steps: %d, loss total: %g, loss validation: %g.\n" % (epoch, step, loss_total, loss_val))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_TEST, MODEL_NAME), global_step)

            # close the multi threads
            coord.request_stop()
            coord.join(threads)
            print("Close the multi threads!")

        return Loss


def backward_linear(Data):
    """Bachward for linear regression

       Args:
           Data: input data consists low and high input and output, for example:[X, Y_]
       Returns:
           loss_total: loss value for every steps
           check point: store the model of the neural network

       Raises:
           Exception: the quantity of input and output are not same
       """

    # data
    graph = tf.Graph()
    X = Data[0]
    Y_ = Data[1]

    if X.shape[0] != Y_.shape[0]:
        raise Exception("The quantity of Input X and Compare Y_ are not same!")

    Loss = []
    with graph.as_default():
        print("This is the process of all the Dose!")
        print("There are %d data in this process." % X.shape[0])
        print("Features of X: %d" % X.shape[1])
        print("Learning rate is: %f" % learning_rate)
        # Init all the parameters
        global_step = tf.Variable(0, trainable=False)

        STEPS = int(Epoch * X.shape[0] / BATCH_SIZE) + 1
        epoch = 0

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, Forward.INPUT_NODE], name='x_Input')
            y_ = tf.placeholder(tf.float32, [None, Forward.OUTPUT_NODE], name='y_Exact')
        y = Forward.forward_linear(x, regularizer=None)

        # lost function
        with tf.name_scope('loss'):
            loss_mse = tf.reduce_mean(tf.square(y - y_))
            loss = loss_mse + tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('loss', loss)

        # Todo
        # LM algorithm

        # learning_rate = tf.train.exponential_decay(
        #     LEARNING_RATE_BASE,
        #     global_step,
        #     X.shape[0] / BATCH_SIZE,
        #     LEARNING_RATE_DECAY,
        #     staircase=True
        # )

        with tf.name_scope('train'):
            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
            # train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step)
            train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99).minimize(loss, global_step)

        # EMA algorithm
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        # ready for storing the model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Get the check point
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH_LINEAR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # begin multi threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            print("Begin the multi threads!")

            # Graph
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./logs_linear/", sess.graph)

            # Training
            for i in range(STEPS):
                start = (i * BATCH_SIZE) % int(X.shape[0])
                end = start + BATCH_SIZE
                # if finish all the data
                if end >= X.shape[0]:
                    end = X.shape[0]

                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: X[start:end], y_: Y_[start:end]})

                if i % 4000 == 0:
                    print("Steps are: %d , loss is: %g." % (step, loss_value))
                    rs = sess.run(merged, feed_dict={x: X[start:end], y_: Y_[start:end]})
                    writer.add_summary(rs, i)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_LINEAR, MODEL_NAME), global_step)

                # a round
                if end == X.shape[0]:
                    # get the results
                    epoch += 1
                    loss_total = sess.run(loss, feed_dict={x: X, y_: Y_})

                    Loss.append(loss_total)
                    # Loss.append(loss_total*10000)
                    print("After %d  epoch(s), steps are: %d, loss total is: %g.\n" % (epoch, step, loss_total))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH_LINEAR, MODEL_NAME), global_step)

            # close the multi threads
            coord.request_stop()
            coord.join(threads)
            print("Close the multi threads!")

        return Loss



def main():

    Data = [X, Y_]


    # k-means
    # X_low, Y_low, X_high, Y_high, _ = GenerateClusterData(Path=path)
    # Data = [X_high, Y_high, X_low, Y_low]
    #
    # # get the line of loss
    # loss_low, loss_high = backward_kmeans(Data)

    # plt.figure('Loss Total')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.plot(loss_low, 'r-', label='loss total low')
    # plt.plot(loss_high, 'b-', label='loss total high')
    # plt.savefig(path_figure)
    # plt.show()


if __name__ == '__main__':
    # 1 image for test
    # init
    X_train, Y_train, maxvalue = get_train_data()
    X_val, Y_val = get_test_data()
    Data = [X_train, Y_train, X_val, Y_val, maxvalue]

    loss_total = backward(Data)
    plt.figure('Loss Total')
    plt.plot(loss_total)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
