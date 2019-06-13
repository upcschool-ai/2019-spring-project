"""
You can find documentation about how to use tf.data and all its classes in the following links:
https://www.tensorflow.org/guide/datasets
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
https://www.tensorflow.org/guide/performance/datasets
"""
import argparse
import csv
import multiprocessing
import os

import tensorflow as tf


def create_dataset(dataset_path, images_dir, num_epochs, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: _generator(dataset_path, images_dir),
                                             output_types=(tf.string, tf.string),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(100)  # Shuffling buffer
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
