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

import numpy as np
import tensorflow as tf

import input_pipeline


NUMBER_CLASSES = 2


def main(dataset_csv, images_dir, num_epochs, batch_size, learning_rate, logdir):
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

    # Model
    # TODO: implement AlexNet

    # Loss and optimizer
    # TODO: include an appropriate loss for the problem and an optimizer to create a training op
    # Parameter suggestion: learning rate ~= 1E-4

    num_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('*'*80)
    print('Num trainable parameters: {!r}'.format(num_trainable_params))
    print('*'*80)

    # Summary writer
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # ----------------- RUN PHASE ------------------- #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                # TODO: run the train step. i.e.: `_, loss = sess.run([train_op, loss_op], feed_dict={...})
                step, x, y = sess.run([global_step, images, labels])
                print('Images shape: {}\tLabels shape: {}'.format(x.shape, y.shape))
                # TODO: print how the loss is evolving per step in order to check if the model is converging
                print('Step {}\tLoss={}'.format(step, None))
        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_csv', help='Path to the CSV decribing the dataset')
    parser.add_argument('images_dir', help='Path to the images directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate')
    args = parser.parse_args()

    main(args.dataset_csv, args.images_dir, args.num_epochs, args.batch_size, args.learning_rate, args.logdir)
