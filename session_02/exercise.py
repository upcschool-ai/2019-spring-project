import argparse
from abc import ABCMeta, abstractmethod


class Operation(object):
    """Base class that holds the interface for operations in our graph"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, *inputs):
        """Compute the operation given the inputs"""
        pass

    @abstractmethod
    def backward(self, *gradients):
        """Compute the outgoing gradient given the flowing gradients.
        Each input from the forward pass must have an output gradient here"""
        pass


class LinearRegressor(Operation):
    def __init__(self):
        # TODO: define attributes
        pass

    def forward(self, *inputs):
        # TODO: compute the forward pass
        pass

    def backward(self, *gradients):
        # TODO: compute the outgoing gradients
        pass


class MSELoss(Operation):

    def forward(self, *predictions):
        # TODO: compute the loss given the predictions
        pass

    def backward(self, *gradients):
        # TODO: compute the gradient for the predictions
        pass


class SGDOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, model, loss):
        pass


def main(learning_rate):
    model = LinearRegressor()
    loss = MSELoss()
    optimizer = SGDOptimizer(learning_rate=learning_rate)

    # TODO: definition phase

    # TODO: run phase


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=10e-3,
                        help='Learning rate for the optimization step')
    args = parser.parse_args()
    main(args.learning_rate)
