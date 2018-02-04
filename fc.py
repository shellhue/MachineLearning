import numpy as np

class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        全链接层初始化

        :param input_size: 输入纬度
        :param output_size: 输出纬度
        :param activator: 激活函数
        '''
        self.w = np.random.uniform(-0.1, 0.1, [output_size, input_size])
        self.w_grad = np.zeros([output_size, input_size])
        self.b = np.zeros([output_size, 1])
        self.b_grad = np.zeros([output_size, 1])
        self.activator = activator
        self.output = np.zeros([output_size, 1])
        self.delta = np.zeros([input_size, 1])
        self.input = np.zeros([input_size, 1])

    def forward(self, input_array):
        self.input = input_array
        z = np.dot(self.w, input_array) + self.b
        self.output = self.activator.forward(z)

    def backward(self, delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(self.w.T, delta_array)
        self.w_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.w += learning_rate * self.w_grad
        self.b += learning_rate * self.b_grad

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1.0 - output)

class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(layers.length - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def forward(self, input_vec):
        for layer in self.layers:
            layer.forward(input_vec)

    def backward(self, label):
        predict = self.layers[-1].output
        delta = self.layers[-1].activator.backward(predict) * (label - predict)
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

def gradient_check(network, sample_feature, sample_label):
    