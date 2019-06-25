import argparse

import tensorflow as tf

import alexnet
import input_pipeline


def main(dataset_csv, images_dir, steps, batch_size, learning_rate, logdir):
    # Input pipeline
    train_input_fn = input_pipeline.create_dataset(dataset_csv, images_dir, None, batch_size)

    # Estimator params
    estimator_params = dict(
        num_classes=2,
        learning_rate=learning_rate
    )

    # Estimator
    estimator = tf.estimator.Estimator(model_fn=alexnet.alexnet, params=estimator_params, model_dir=logdir)

    # Train
    estimator.train(train_input_fn, max_steps=steps)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_csv', help='Path to the CSV decribing the dataset')
    parser.add_argument('images_dir', help='Path to the images directory')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-s', '--steps', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate')
    args = parser.parse_args()

    main(args.dataset_csv, args.images_dir, args.steps, args.batch_size, args.learning_rate, args.logdir)
