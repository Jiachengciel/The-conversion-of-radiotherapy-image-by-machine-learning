import tensorflow as tf
import numpy as np
from Input_data import getMaxvalue, getMinvalue

INPUT_NODE = 7
HIDDEN_LAYER1_NODE = 30
HIDDEN_LAYER2_NODE = 5
OUTPUT_NODE = 1


def get_weight(shape, regularizer):
    """Get the weight correspondent the number of nodes

    Args:
        shape: shape of the weight
        regularizer: the number of the regularizer if you want to add

    Returns:
        w: array of the weight
    """

    w = tf.Variable(np.random.randn(shape[0], shape[1]), dtype=tf.float32) / np.sqrt(shape[0])
    # w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
    if regularizer is not None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    """Get the bias correspondent the number of nodes

    Args:
        shape: shape of the bias

    Returns:
        b: array of the bias
    """

    b = tf.Variable(tf.zeros(shape), name='bias')
    return b


def forward(x, regularizer, maxvalue, Is_model_high=True):
    """Construct the forward process of Neural Network

    Args:
        x : input of Neural Network
        regularizer : the number of the regularizer if you want to add
        maxvalue: max value of input image
        Is_model_high: defin whether it is the high dose

    Returns:
        y : output of Neural Network, after calculation
    """

    with tf.name_scope('layer_1'):
        with tf.name_scope('weight_1'):
            w1 = get_weight([INPUT_NODE, HIDDEN_LAYER1_NODE], regularizer)
            tf.summary.histogram(name='layer_1/weights', values=w1)
        with tf.name_scope('bias_1'):
            b1 = get_bias([HIDDEN_LAYER1_NODE])
            tf.summary.histogram(name='layer_1/bias', values=b1)
        with tf.name_scope('out_1'):
            y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
            tf.summary.histogram(name='layer_1/outputs', values=y1)

    with tf.name_scope('layer_2'):
        with tf.name_scope('weight_2'):
            w2 = get_weight([HIDDEN_LAYER1_NODE, HIDDEN_LAYER2_NODE], regularizer)
            tf.summary.histogram(name='layer_2/weights', values=w2)
        with tf.name_scope('bias_2'):
            b2 = get_bias([HIDDEN_LAYER2_NODE])
            tf.summary.histogram(name='layer_2/bias', values=b2)
        with tf.name_scope('out_2'):
            y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
            tf.summary.histogram(name='layer_2/outputs', values=y2)

    with tf.name_scope('layer_3'):
        with tf.name_scope('weight_3'):
            w3 = get_weight([HIDDEN_LAYER2_NODE, OUTPUT_NODE], regularizer)
            tf.summary.histogram(name='layer_3/weights', values=w3)
        with tf.name_scope('bias_3'):
            b3 = get_bias([OUTPUT_NODE])
            tf.summary.histogram(name='layer_3/bias', values=b3)
        with tf.name_scope('out_3'):
            y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
            tf.summary.histogram(name='layer_3/outputs', values=y3)

    with tf.name_scope('layer_4'):
        with tf.name_scope('weight_4'):
            w4 = tf.Variable(150, dtype=tf.float32)
            tf.summary.histogram(name='layer_4/weights', values=w4)
        with tf.name_scope('bias_4'):
            b4 = tf.Variable(0, dtype=tf.float32)
            tf.summary.histogram(name='layer_4/bias', values=b4)
        with tf.name_scope('out'):
            if Is_model_high:
                max_value = tf.constant(maxvalue, dtype=tf.float32)
                y = y3 * (w4*max_value) + b4
            else:
                y = y3 * w4 + b4
            tf.summary.histogram(name='layer_4/outputs', values=y)

    return y


def forward_linear(x, regularizer):
    with tf.name_scope('linear'):
        with tf.name_scope('weight'):
            w = get_weight([INPUT_NODE, OUTPUT_NODE], regularizer)
            tf.summary.histogram(name='linear/weights', values=w)
        with tf.name_scope('bias'):
            b = get_bias([OUTPUT_NODE])
            tf.summary.histogram(name='linear/bias', values=b)
        with tf.name_scope('out'):
            y = tf.add(tf.matmul(x, w), b)
            tf.summary.histogram(name='linear/outputs', values=y)

    return y
