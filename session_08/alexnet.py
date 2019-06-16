"""
Implement AlexNet: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Useful links:
https://www.tensorflow.org/api_docs/python/tf/nn
https://www.tensorflow.org/api_docs/python/tf/layers
https://www.tensorflow.org/api_docs/python/tf/keras/layers
"""
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import input_pipeline


def main(dataset_csv, dataset_dir, num_epochs, batch_size, logdir):
    # ----------------- TRAINING LOOP SETUP ---------------- #
    logdir = os.path.expanduser(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # ----------------- DEFINITION PHASE ------------------- #
    global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)

    # Input pipeline
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            dataset = input_pipeline.create_dataset(dataset_csv, dataset_dir, num_epochs, batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

    # Model
    conv1 = conv_layer(images, filters=96, kernel_size=(11, 11), strides=(4, 4), lrn=True, max_pool=True, scope='conv1')
    conv2 = conv_layer(conv1, filters=256, kernel_size=(5, 5), lrn=True, max_pool=True, scope='conv2')
    conv3 = conv_layer(conv2, filters=385, kernel_size=(3, 3), scope='conv3')
    conv4 = conv_layer(conv3, filters=385, kernel_size=(3, 3), scope='conv4')
    conv5 = conv_layer(conv4, filters=256, kernel_size=(3, 3), max_pool=True, scope='conv5')

    fc1 = fully_connected(conv5, units=4096, dropout=0.5, scope='fc1')
    fc2 = fully_connected(fc1, units=4096, dropout=0.5, scope='fc2')
    logits = fully_connected(fc2, units=1000, scope='fc3')

    # Loss
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    # Optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
    train_step = optimizer.minimize(loss_op, global_step=global_step)

    # ----------------- RUN PHASE ------------------- #
    with tf.Session() as sess:
        try:
            while True:
                # Run the train step
                _, loss, step = sess.run([train_step, loss_op, global_step])
                # Print how the loss is evolving per step in order to check if the model is converging
                print('Step {}\tLoss={}'.format(step, loss))
        except tf.errors.OutOfRangeError:
            pass


def conv_layer(inputs, filters, kernel_size, strides=(1, 1), lrn=False, max_pool=False, scope='conv_layer'):
    with tf.variable_scope(scope):
        output = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                  activation=tf.nn.relu)
        if lrn:
            output = tf.nn.lrn(output, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
        if max_pool:
            output = tf.layers.max_pooling2d(output, pool_size=(3, 3), strides=(2, 2))
        return output


def fully_connected(inputs, units, activation=tf.nn.relu, dropout=None, scope='fully_connected'):
    with tf.variable_scope(scope):
        output = tf.layers.dense(inputs, units=units, activation=activation)
        if dropout:
            output = tf.nn.dropout(output, keep_prob=dropout)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlexNet model training')
    parser.add_argument('dataset_csv', help='Path to the CSV decribing the dataset')
    parser.add_argument('dataset_dir', help='Directory where the csv and the images folder are located')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    main(args.dataset_csv, args.dataset_dir, args.num_epochs, args.batch_size, args.logdir)