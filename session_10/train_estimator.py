import argparse


def main(dataset_csv, images_dir, steps, batch_size, learning_rate, logdir):
    # TODO: [Exercise VIII] 3. Import input_fn from input_pipeline & model_fn from alexnet
    # TODO: [Exercise VIII] 4. Create tf.estimator.Estimator
    # TODO: [Exercise VIII] 5. Train estimator
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
