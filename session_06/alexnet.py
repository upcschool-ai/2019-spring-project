"""
Implement AlexNet: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Useful links:
https://www.tensorflow.org/api_docs/python/tf/nn
https://www.tensorflow.org/api_docs/python/tf/layers
https://www.tensorflow.org/api_docs/python/tf/keras/layers
"""
import argparse
import os

import tensorflow as tf

import input_pipeline


def main(dataset_path, images_dir, num_epochs, batch_size, logdir):
    # ----------------- DEFINITION PHASE ------------------- #
    # Input pipeline
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            dataset = input_pipeline.create_dataset(dataset_path, images_dir, num_epochs, batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

    # Model definition
    # TODO: implement AlexNet

    # ----------------- RUN PHASE ------------------- #
    with tf.Session() as sess:
        try:
            while True:
                # TODO: run the train step. i.e.: `_, loss = sess.run([train_op, loss_op], feed_dict={...})
                pass
        except tf.errors.OutOfRangeError:
            pass

    logdir = os.path.expanduser(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_path', help='Path to dataset description')
    parser.add_argument('images_dir', help='Image directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    main(args.dataset_path, args.images_dir, args.num_epochs, args.batch_size, args.logdir)
