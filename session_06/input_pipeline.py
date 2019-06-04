"""
You can find documentation about how to use tf.data and all its classes in the following links:
https://www.tensorflow.org/guide/datasets
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
https://www.tensorflow.org/guide/performance/datasets
"""
import argparse
import os

import tensorflow as tf


def create_dataset(dataset_path, images_dir, num_epochs, batch_size):
    # TODO: create and return a tf.data.Dataset object holding the ImageNet input pipeline
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_path', help='Path to dataset description')
    parser.add_argument('images_dir', help='Image directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            dataset = create_dataset(args.dataset_path, args.images_dir, args.num_epochs, args.batch_size)
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                images, labels = sess.run(batch)
                print images.shape, labels.shape
        except tf.errors.OutOfRangeError:
            pass

    logdir = os.path.expanduser(args.logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
