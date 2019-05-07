import numpy as np


class SampleGenerator(object):
    def __init__(self):
        self.W = np.random.uniform(-5, 5)  # Set the weight to be a random number between [-5, 5]
        self.b = np.random.uniform(-5, 5)  # Set the bias to be a random number between [0, 5]

    def __call__(self, num_iters):
        for i in xrange(num_iters):
            x = np.random.uniform(-200, 200)  # Draw random input from [-100, 100]
            y = self.W * x + self.b  # Compute output for current input
            yield x, y


def sample_generator(num_iters):
    """Generator for samples of the "hidden" data distribution

    :param num_iters:
    :return: tuples of (x, y)
    """
    return SampleGenerator()(num_iters)
