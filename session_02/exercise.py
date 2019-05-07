import argparse
from abc import ABCMeta, abstractmethod


class Operation(object):
    """
    Base class that holds the interface for operations in our graph.
    As this is a toy example, it's only defined for N:1 operations (multiple input, single output)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, *inputs):
        """Compute the operation given the inputs"""
        pass

    @abstractmethod
    def backward(self, gradient):
        """Compute the outgoing gradient and any intermediate one given the flowing gradients"""
        pass


class LinearRegressor(Operation):
    def __init__(self):
        # TODO: define attributes
        pass

    def forward(self, *inputs):
        # TODO: compute the forward pass
        pass

    def backward(self, gradient):
        # TODO: compute the outgoing gradients
        pass


class MSELoss(Operation):

    def forward(self, *predictions):
        # TODO: compute the loss given the predictions
        pass

    def backward(self, gradient):
        # TODO: compute the gradient for the predictions
        pass


class SGDOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, model, loss):
        pass


def main(learning_rate):
    # TODO: definition phase

    # TODO: run phase
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=10e-6,
                        help='Learning rate for the optimization step')
    args = parser.parse_args()
    main(args.learning_rate)
