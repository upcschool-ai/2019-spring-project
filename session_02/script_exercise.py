import argparse
import os

import tensorflow as tf

import aidl


def main(learning_rate, logdir):
    """

    :param learning_rate:
    :param logdir:
    :return:
    """

    """
            DEFINITION PHASE
    """
    graph = tf.Graph()
    with graph.as_default():
        dataset = aidl.SampleGenerator()

        # Inputs and labels
        x = tf.placeholder(dtype=tf.float32, name='x')
        y = tf.placeholder(dtype=tf.float32, name='y')

        # ------------------------ FORWARD PASS -----------------------------------
        # Linear regression forward pass
        W = tf.get_variable('W', shape=[], dtype=tf.float32)
        b = tf.get_variable('b', shape=[], dtype=tf.float32)
        z = W * x
        pred = z + b

        # Compute loss
        diff = pred - y
        loss = tf.pow(diff, 2)

        # ------------------------ BACKWARD PASS -----------------------------------
        # Loss backprop
        loss_grad = 2 * diff

        # Linear regression backprop
        W_grad = x * loss_grad
        b_grad = 1 * loss_grad
        x_grad = W * loss_grad

        # ------------------------ OPTIMIZATION -----------------------------------
        W_update = W.assign(W - learning_rate * W_grad)
        b_update = b.assign(b - learning_rate * b_grad)
        train_op = tf.group(W_update, b_update)

    """
                RUN PHASE
    """
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for i, (input_sample, gt_value) in enumerate(dataset(5000), start=1):
            _, loss_value = sess.run([train_op, loss], feed_dict={x: input_sample, y: gt_value})
            print 'Loss[{}]: {}'.format(i, loss_value)
        W_pred, b_pred = sess.run([W, b])
        print 'W GT: {}. W pred: {}'.format(dataset.W, W_pred)
        print 'b GT: {}. b pred: {}'.format(dataset.b, b_pred)

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, graph=graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=10e-6,
                        help='Learning rate for the optimization step')
    parser.add_argument('-l', '--logdir', default='~/tmp/aidl', help='Log dir for tfevents')
    args = parser.parse_args()
    main(args.learning_rate, os.path.expanduser(args.logdir))
