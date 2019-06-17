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


NUMBER_CLASSES = 2


def main(dataset_csv, images_dir, num_epochs, batch_size, logdir):
    # ----------------- TRAINING LOOP SETUP ---------------- #
    logdir = os.path.expanduser(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # ----------------- DEFINITION PHASE ------------------- #
    global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)

    # Input pipeline
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            dataset = input_pipeline.create_dataset(dataset_csv, images_dir, num_epochs, batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

            tf.summary.image('input', images, max_outputs=batch_size)

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

    flat = tf.reshape(conv5, [-1, 5*5*256])
    fc1 = fully_connected(flat, units=4096, dropout_rate=0, scope='fc1')
    tf.summary.histogram('fc1', fc1)
    fc2 = fully_connected(fc1, units=4096, dropout_rate=0, scope='fc2')
    tf.summary.histogram('fc2', fc2)
    logits = fully_connected(fc2, activation=None, units=NUMBER_CLASSES, scope='fc3')
    tf.summary.histogram('logits', logits)
    softmax = tf.nn.softmax(logits, name='softmax')

    # Loss
    loss_op = tf.losses.softmax_cross_entropy(labels, logits)
    tf.summary.scalar('loss', loss_op)

    # Optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
    train_step = optimizer.minimize(loss_op, global_step=global_step)

    # Summary writer
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    summary_op = tf.summary.merge_all()

    # ----------------- RUN PHASE ------------------- #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                # Run the train step
                _, loss, step, summ_val, logits_val, labels_val, softmax_val = sess.run([train_step, loss_op, global_step, summary_op, logits, labels, softmax])
                # Print how the loss is evolving per step in order to check if the model is converging
                print('Step {}\tLoss={}'.format(step, loss))
                # print('Logits: {}'.format(logits_val))
                # print('Softmax: {}'.format(softmax_val))
                # print('Labels: {}'.format(labels_val))
                writer.add_summary(summ_val, global_step=step)
        except tf.errors.OutOfRangeError:
            pass


def conv_layer(inputs, filters, kernel_size, strides=(1, 1), lrn=False, max_pool=False, padding='valid', scope='conv_layer'):
    with tf.variable_scope(scope):
        output = tf.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, activation=tf.nn.relu, padding=padding,
            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
            bias_initializer=tf.initializers.ones(),
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
        )(inputs)
        if dropout_rate:
            output = tf.layers.Dropout(rate=dropout_rate)(output, training=True)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_csv', help='Path to the CSV decribing the dataset')
    parser.add_argument('images_dir', help='Path to the images directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    main(args.dataset_csv, args.images_dir, args.num_epochs, args.batch_size, args.logdir)
