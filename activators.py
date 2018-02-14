import numpy as np


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1.0 - output)


class IdentityActivator(object):
    def forward(self, weighted_input):
        return np.array(weighted_input)

    def backward(self, output):
        return 1

