import numpy as np
from functools import reduce


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


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, input_activator, output_activator):
        '''
        全链接层初始化

        :param input_size: 输入纬度
        :param output_size: 输出纬度
        :param activator: 激活函数
        '''
        self.w = np.random.uniform(-0.1, 0.1, [output_size, input_size])
        self.w_grad = np.zeros([output_size, input_size])
        self.b = np.zeros(output_size)
        self.b_grad = np.zeros(output_size)
        self.input_activator = input_activator
        self.output_activator = output_activator
        self.output = np.zeros(output_size)
        self.delta = np.zeros(input_size)
        self.input = np.zeros(input_size)

    def forward(self, input_array):
        self.input = input_array
        z = np.dot(self.w, input_array) + self.b
        self.output = self.output_activator.forward(z)

    def backward(self, delta_array):
        self.delta = self.input_activator.backward(self.input) * np.dot(self.w.T, delta_array)
        self.w_grad = np.dot(delta_array[:, None], self.input[None, :])
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.w += learning_rate * self.w_grad
        self.b += learning_rate * self.b_grad


class Network(object):
    def __init__(self, layers, output_activator):
        self.layers = []
        for i in range(layers.length - 1):
            if i == layers.length - 2:
                self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator(), output_activator))
            else:
                self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator(), SigmoidActivator()))

    def forward(self, input_vec):
        for layer in self.layers:
            layer.forward(input_vec)

    def backward(self, label):
        predict = self.layers[-1].output
        delta = self.layers[-1].output_activator.backward(predict) * (label - predict)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, input_vecs, labels, rate, epoch):
        for _ in range(epoch):
            for (input_vec, label) in zip(input_vecs, labels):
                self.train_one_example(input_vec, label, rate)

    def train_one_example(self, input_vec, label, rate):
        self.forward(input_vec)
        self.backward(label)
        self.update(rate)

    def predict(self, input_vec):
        self.forward(input_vec)
        return self.layers[-1].output


def single_fc_layer_gradient_check():
    error_fuction = lambda output: output.sum()
    input = np.array([1, 2, 3, 4, 4])
    delta = np.ones(3)
    layer = FullConnectedLayer(5, 3, SigmoidActivator(), IdentityActivator())
    layer.forward(input)
    layer.backward(delta)
    epsilon = 0.0001

    for i in range(layer.w.shape[0]):
        for j in range(layer.w.shape[1]):
            layer.w[i][j] += epsilon
            layer.forward(input)
            error1 = error_fuction(layer.output)
            layer.w[i][j] -= 2 * epsilon
            layer.forward(input)
            error2 = error_fuction(layer.output)
            expected_gradient = (error1 - error2) / (2 * epsilon)
            print('expected gradient: %f, actual: %f' % (expected_gradient, layer.w_grad[i][j]))


def gradient_check(network, sample_feature, sample_label):
    layer_index = np.random.randint(0, network.layers.length)
    layer = network.layers[layer_index]
    network.forward(sample_feature)
    network.backward(sample_label)

    def network_error(predict, label):
        minus = list(map(lambda item: item[0] - item[1], list(zip(label, predict))))
        return reduce(lambda a, b: a + b, minus, 0)

    epsilon = 0.0001
    output_index = np.random.randint(0, layer.output_size)
    input_index = np.random.randint(0, layer.input_size)
    actual_gradient = layer.w_grad[output_index][input_index]

    layer.w[output_index][input_index] += epsilon
    error1 = network_error(network.predict(sample_feature), sample_label)
    layer.w[output_index][input_index] -= epsilon * 2
    error2 = network_error(network.predict(sample_feature), sample_label)
    expected_gradient = (error1 - error2) / (2 * epsilon)
    print('expected gradient: \t%f\nactual gradient:\t%f\n' % (expected_gradient, actual_gradient))


if __name__ == '__main__':
    single_fc_layer_gradient_check()