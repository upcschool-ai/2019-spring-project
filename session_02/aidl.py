import numpy as np


class SampleGenerator(object):
    def __init__(self):
        self.W = np.random.uniform(-10, 10)  # Set the weight to be a random number between [-10, 10]
        self.b = np.random.uniform(-5, 5)  # Set the bias to be a random number between [0, 5]

    def __call__(self, num_iters, batch_size=1):
        for i in xrange(num_iters):
            x = np.random.uniform(-100, 100, size=batch_size)  # Draw random input from [-100, 100]
            y = self.W * x + self.b  # Compute output for current input
            yield x, y


def sample_generator(num_iters, batch_size=1):
    """Generator for samples of the "hidden" data distribution

    :param num_iters:
    :param batch_size:
    :return: tuples of (x, y). If batch_size > 1, x and y are arrays with batch dimension
    """
    return SampleGenerator()(num_iters, batch_size=batch_size)
