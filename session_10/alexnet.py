"""
Implement AlexNet: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Useful links:
https://www.tensorflow.org/api_docs/python/tf/nn
https://www.tensorflow.org/api_docs/python/tf/layers
https://www.tensorflow.org/api_docs/python/tf/keras/layers
"""
from __future__ import print_function

import tensorflow as tf


def alexnet(images, labels, mode, params):
    # Parameters
    num_classes = params.get('num_classes', 2)
    learning_rate = params.get('learning_rate', 1e-5)

    # Model
    print(images.get_shape().as_list())
    conv1 = conv_layer(images, filters=96, kernel_size=(11, 11), strides=(4, 4), lrn=True, max_pool=True, scope='conv1')
    print(conv1.get_shape().as_list())
    tf.summary.histogram('conv1', conv1)
    conv2 = conv_layer(conv1, filters=256, kernel_size=(5, 5), lrn=True, max_pool=True, padding='same', scope='conv2')
    print(conv2.get_shape().as_list())
    tf.summary.histogram('conv2', conv2)
    conv3 = conv_layer(conv2, filters=384, kernel_size=(3, 3), padding='same', scope='conv3')
    print(conv3.get_shape().as_list())
    tf.summary.histogram('conv3', conv3)
    conv4 = conv_layer(conv3, filters=384, kernel_size=(3, 3), padding='same', scope='conv4')
    print(conv4.get_shape().as_list())
    tf.summary.histogram('conv4', conv4)
    conv5 = conv_layer(conv4, filters=256, kernel_size=(3, 3), max_pool=True, scope='conv5')
    tf.summary.histogram('conv5', conv5)
    print(conv5.get_shape().as_list())

    flat = tf.reshape(conv5, [-1, 5 * 5 * 256])
    fc1 = fully_connected(flat, units=4096, dropout_rate=0.5, scope='fc1')
    tf.summary.histogram('fc1', fc1)
    fc2 = fully_connected(fc1, units=4096, dropout_rate=0.5, scope='fc2')
    tf.summary.histogram('fc2', fc2)
    logits = fully_connected(fc2, activation=None, units=num_classes, scope='fc3')
    tf.summary.histogram('logits', logits)
    softmax = tf.nn.softmax(logits, name='softmax')

    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': softmax}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Loss
    xe_loss_op = tf.losses.softmax_cross_entropy(labels, logits)
    loss_op = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss_op)

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss_op, eval_metric_ops=[])

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss_op, train_op=train_op)


def conv_layer(inputs, filters, kernel_size, strides=(1, 1), lrn=False, max_pool=False, padding='valid',
               scope='conv_layer'):
    with tf.variable_scope(scope):
        output = tf.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, activation=tf.nn.relu, padding=padding,
            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
            bias_initializer=tf.initializers.ones(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(inputs)
        if lrn:
            output = tf.nn.lrn(output, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
        if max_pool:
            output = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        return output


def fully_connected(inputs, units, activation=tf.nn.relu, dropout_rate=None, scope='fully_connected'):
    with tf.variable_scope(scope):
        output = tf.layers.Dense(
            units=units, activation=activation,
            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
            bias_initializer=tf.initializers.ones(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(inputs)
        if dropout_rate:
            output = tf.layers.Dropout(rate=dropout_rate)(output, training=True)

        return output
