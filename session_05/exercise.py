import argparse
import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf

import aidl


class Operation(object):
    """
    Base class that holds the interface for operations in our graph.
    As this is a toy example, it's only defined for N:1 operations (multiple input, single output)
    """
    __metaclass__ = ABCMeta

    def __init__(self, scope=None):
        with tf.variable_scope(scope, default_name=self.__class__.__name__) as captured_scope:
            self._scope = captured_scope

    @abstractmethod
    def forward(self, *inputs):
        """Compute the operation given the inputs"""
        pass


class LinearRegressor(Operation):
    WEIGHT = 'W'
    BIAS = 'b'
    INPUT = 'input'

    def __init__(self, scope=None):
        super(LinearRegressor, self).__init__(scope=scope)
        with tf.variable_scope(self._scope):
            self.W = tf.get_variable(LinearRegressor.WEIGHT, shape=[], dtype=tf.float32)
            self.b = tf.get_variable(LinearRegressor.BIAS, shape=[], dtype=tf.float32)

    @property
    def trainable_variables(self):
        return {LinearRegressor.WEIGHT: self.W, LinearRegressor.BIAS: self.b}

    def forward(self, *inputs):
        with tf.name_scope(self._scope.original_name_scope):
            input_tensor = inputs[0]
            # Linear regression operations
            z = self.W * input_tensor
            y = z + self.b

            # Return estimation of y
            return y


class MSELoss(Operation):
    PREDICTION = 'prediction'

    def forward(self, *inputs):
        with tf.name_scope(self._scope.original_name_scope):
            prediction, gt = inputs
            # Compute the loss
            diff = prediction - gt
            loss = tf.pow(diff, 2)
            return loss


def main(learning_rate, logdir):
    graph = tf.Graph()
    with graph.as_default():
        dataset = aidl.SampleGenerator()
        model = LinearRegressor()
        loss = MSELoss()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        x = tf.placeholder(dtype=tf.float32, name='x')
        y = tf.placeholder(dtype=tf.float32, name='y')
        prediction = model.forward(x)
        loss_op = loss.forward(prediction, y)
        train_op = optimizer.minimize(loss_op)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for i, (input_sample, gt_value) in enumerate(dataset(5000), start=1):
            _, loss_value = sess.run([train_op, loss_op], feed_dict={x: input_sample, y: gt_value})
            print 'Loss[{}]: {}'.format(i, loss_value)
        W_pred, b_pred = sess.run([model.W, model.b])
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
