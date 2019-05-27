import argparse
from abc import ABCMeta, abstractmethod

import tensorflow as tf

import aidl


class Operation(object):
    """
    Base class that holds the interface for operations in our graph.
    As this is a toy example, it's only defined for N:1 operations (multiple input, single output)
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._local_derivatives = {}
        self._gradients = {}

    def add_local_derivative(self, name, value):
        self._local_derivatives[name] = value

    def get_local_derivative(self, name):
        return self._local_derivatives[name]

    def add_gradient(self, name, value):
        self._gradients[name] = value

    def get_gradient(self, name):
        return self._gradients[name]

    @abstractmethod
    def forward(self, *inputs):
        """Compute the operation given the inputs"""
        pass

    @abstractmethod
    def backward(self, gradient):
        """Compute the outgoing gradient and any intermediate one given the flowing gradients"""
        pass


class LinearRegressor(Operation):
    WEIGHT = 'W'
    BIAS = 'b'
    INPUT = 'input'

    def __init__(self):
        super(LinearRegressor, self).__init__()
        self.W = tf.get_variable(LinearRegressor.WEIGHT, shape=[], dtype=tf.float32)
        self.b = tf.get_variable(LinearRegressor.BIAS, shape=[], dtype=tf.float32)

    @property
    def trainable_variables(self):
        return {LinearRegressor.WEIGHT: self.W, LinearRegressor.BIAS: self.b}

    def forward(self, *inputs):
        input_tensor = inputs[0]
        # Linear regression operations
        z = self.W * input_tensor
        y = z + self.b

        # Compute local derivatives
        self.add_local_derivative(LinearRegressor.WEIGHT, input_tensor)
        self.add_local_derivative(LinearRegressor.BIAS, 1)
        self.add_local_derivative(LinearRegressor.INPUT, self.W)

        # Return estimation of y
        return y

    def backward(self, gradient):
        for tensor_name, local_derivative in self._local_derivatives.iteritems():
            tensor_gradient = gradient * local_derivative
            self.add_gradient(tensor_name, tensor_gradient)
        return self.get_gradient(LinearRegressor.INPUT)


class MSELoss(Operation):
    PREDICTION = 'prediction'

    def forward(self, *inputs):
        prediction, gt = inputs
        # Compute the loss
        diff = prediction - gt
        loss = tf.pow(diff, 2)
        # Compute the local derivative
        local_derivative = 2 * diff
        self.add_local_derivative(MSELoss.PREDICTION, local_derivative)
        return loss

    def backward(self, gradient):
        out_gradient = gradient * self.get_local_derivative(MSELoss.PREDICTION)
        self.add_gradient(MSELoss.PREDICTION, out_gradient)
        return out_gradient


class SGDOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, model, loss):
        loss_gradient = loss.backward(1)
        _ = model.backward(loss_gradient)
        trainable_vars = model.trainable_variables

        optimize_ops = []
        for var_name, trainable_var in trainable_vars.iteritems():
            optimization_step = -1 * self.learning_rate * model.get_gradient(var_name)
            optimize_op = trainable_var.assign_add(optimization_step)
            optimize_ops.append(optimize_op)
        return tf.group(optimize_ops)


def main(learning_rate, logdir):
    graph = tf.Graph()
    with graph.as_default():
        dataset = aidl.SampleGenerator()
        model = LinearRegressor()
        loss = MSELoss()
        optimizer = SGDOptimizer(learning_rate=learning_rate)

        x = tf.placeholder(dtype=tf.float32, name='x')
        y = tf.placeholder(dtype=tf.float32, name='y')
        prediction = model.forward(x)
        loss_op = loss.forward(prediction, y)
        train_op = optimizer.optimize(model, loss)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for i, (input_sample, gt_value) in enumerate(dataset(5000), start=1):
            _, loss_value = sess.run([train_op, loss_op], feed_dict={x: input_sample, y: gt_value})
            print 'Loss[{}]: {}'.format(i, loss_value)
        W_pred, b_pred = sess.run([model.W, model.b])
        print 'W GT: {}. W pred: {}'.format(dataset.W, W_pred)
        print 'b GT: {}. b pred: {}'.format(dataset.b, b_pred)

    writer = tf.summary.FileWriter(logdir, graph=graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=10e-6,
                        help='Learning rate for the optimization step')
    parser.add_argument('-l', '--logdir', help='Log dir for tfevents')
    args = parser.parse_args()
    main(args.learning_rate, args.logdir)
