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
