"""
You can find documentation about how to use tf.data and all its classes in the following links:
https://www.tensorflow.org/guide/datasets
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
https://www.tensorflow.org/guide/performance/datasets
"""
from __future__ import print_function

import argparse
import csv
import itertools
import multiprocessing
import os

import tensorflow as tf


def create_dataset(dataset_csv, images_dir, num_epochs, batch_size):
    # TODO: [Exercise VIII] 1. Create dataset wrapper function
    dataset = tf.data.Dataset.from_generator(lambda: _generator(dataset_csv, images_dir),
                                             output_types=(tf.string, tf.string),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(500)  # Shuffling buffer
    dataset = dataset.map(_create_sample, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(10)  # Pipelining
    return dataset


def _generator(path, images_dir):
    with open(path) as f:
        reader = csv.reader(f)
        for label, image_path in reader:
            image_path = os.path.join(images_dir, image_path)
            yield image_path, label


def _create_sample(image_path, label):
    with tf.name_scope('create_sample'):
        with tf.name_scope('read_image'):
            raw_image = tf.read_file(image_path)
            image = tf.image.decode_jpeg(raw_image, channels=3)
            label = tf.strings.to_number(label, out_type=tf.int32)
            label = tf.one_hot(label, depth=2)

        with tf.name_scope('preprocessing'):
            mean_channel = [123.68, 116.779, 103.939]
            image = tf.cast(image, dtype=tf.float32)
            image = tf.subtract(image, mean_channel, name='mean_substraction')
            image = tf.image.resize(image, size=(256, 256))

        with tf.name_scope('data_augmentation'):
            image = tf.image.random_crop(image, size=(227, 227, 3))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=20)

    return image, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('train_dataset', help='Path to the CSV decribing the dataset')
    parser.add_argument('val_dataset', help='Path to the CSV decribing the dataset')
    parser.add_argument('images_dir', help='Path to the images directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl/logs', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-v', '--val_iters', type=int, default=10, help='Steps from validation to validation')

    args = parser.parse_args()

    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            with tf.name_scope('training_dataset'):
                train_dataset = create_dataset(args.train_dataset, args.images_dir, args.num_epochs, args.batch_size)
                train_iterator = train_dataset.make_one_shot_iterator()
            with tf.name_scope('validation_dataset'):
                val_dataset = create_dataset(args.val_dataset, args.images_dir, 1, args.batch_size)
                val_iterator = val_dataset.make_initializable_iterator()

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)
            batch = iterator.get_next()

    with tf.Session() as sess:
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        try:
            for train_step in itertools.count(start=1, step=1):
                images, labels = sess.run(batch, feed_dict={handle: train_handle})
                print('TRAINING [Step={}] Images shape: {}\tLabels shape: {}'.format(train_step, images.shape,
                                                                                     labels.shape))

                # Validation
                if train_step % args.val_iters == 0:
                    try:
                        sess.run(val_iterator.initializer)
                        for val_step in itertools.count(start=1, step=1):
                            images, labels = sess.run(batch, feed_dict={handle: val_handle})
                            print(
                                'VALIDATION [Step={}] Images shape: {}\tLabels shape: {}'.format(val_step, images.shape,
                                                                                                 labels.shape))
                    except tf.errors.OutOfRangeError:
                        pass
        except tf.errors.OutOfRangeError:
            pass

    logdir = os.path.expanduser(args.logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
