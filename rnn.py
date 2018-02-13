import numpy as np
from cnn import element_wise_op
from fc import IdentityActivator
from functools import reduce


class ReluActivator(object):
    def forward(self, x):
        return x if x > 0 else 0

    def backward(self, output):
        return 1 if output > 0 else 0


class RecurrentLayer(object):
    def __init__(self, input_size, output_size, input_activator, output_activator):
        self.input_size = input_size
        self.output_size = output_size
        self.input_activator = input_activator
        self.output_activator = output_activator
        self.w = np.random.uniform(-1e-4, 1e-4, [self.output_size, self.input_size])
        self.u = np.random.uniform(-1e-4, 1e-4, [self.output_size, self.output_size])
        self.times = 0

        self.cell_state_list = []
        self.input_list = []

        self.cell_state_list.append(np.zeros([self.output_size, 1]))
        self.input_list.append(np.zeros([self.input_size, 1]))

        self.w_grad = np.zeros(self.w.shape)
        self.u_grad = np.zeros(self.u.shape)

    def forward(self, input_array):
        self.input_list.append(input_array)
        self.times += 1
        current_cell_state = np.dot(self.u, self.cell_state_list[-1]) + np.dot(self.w, input_array)
        element_wise_op(current_cell_state, self.output_activator.forward)
        self.cell_state_list.append(current_cell_state)

    def backward(self, sensitivity_array):
        # 计算delta数组
        delta_list = []
        for i in range(self.times):
            delta_list.append(np.zeros([self.output_size, 1]))
        delta_list.append(sensitivity_array)
        for t in range(self.times - 1, 0, -1):
            state_t = self.cell_state_list[t].copy()
            element_wise_op(state_t, self.output_activator.backward)
            delta_t = np.dot(self.u.T, delta_list[t + 1]) * state_t
            delta_list[t] = delta_t

        # 计算梯度
        u_grad_list = []
        w_grad_list = []
        for t in range(self.times, 0, -1):
            delta_t = delta_list[t]
            previous_cell_state = self.cell_state_list[t - 1]
            input_t = self.input_list[t]
            u_grad_t = np.dot(delta_t, previous_cell_state.T)
            w_grad_t = np.dot(delta_t, input_t.T)
            u_grad_list.append(u_grad_t)
            w_grad_list.append(w_grad_t)

        self.w_grad = reduce(lambda m, n: n + m, w_grad_list, np.zeros(self.w_grad.shape))
        self.u_grad = reduce(lambda m, n: n + m, u_grad_list, np.zeros(self.u_grad.shape))

    def update(self, learning_rate):
        self.w -= self.w_grad * learning_rate
        self.u -= self.u_grad * learning_rate

    def reset_state(self):
        self.times = 0
        self.cell_state_list = []
        self.input_list = []
        self.cell_state_list.append(np.zeros([self.output_size, 1]))
        self.input_list.append(np.zeros([self.input_size, 1]))


def rnn_gradient_check():
    error_function = lambda x: x.sum()
    x1 = np.array([[2], [4], [10], [10], [10]])
    x2 = np.array([[8], [6], [5], [10], [10]])
    x3 = np.array([[18], [13], [15], [10], [10]])
    delta_array = np.array([[1], [1], [1]])

    rnn = RecurrentLayer(5, 3, ReluActivator(), IdentityActivator())
    rnn.forward(x1)
    rnn.forward(x2)
    rnn.forward(x3)
    rnn.backward(delta_array)

    epsilon = 0.0001
    for i in range(rnn.u.shape[0]):
        for j in range(rnn.u.shape[1]):
            rnn.reset_state()
            rnn.u[i, j] += epsilon
            rnn.forward(x1)
            rnn.forward(x2)
            rnn.forward(x3)
            err1 = error_function(rnn.cell_state_list[-1])
            rnn.reset_state()
            rnn.u[i, j] -= epsilon * 2
            rnn.forward(x1)
            rnn.forward(x2)
            rnn.forward(x3)
            err2 = error_function(rnn.cell_state_list[-1])
            expected_grad = (err1 - err2) / (2 * epsilon)
            rnn.u[i, j] += epsilon
            print('expected u grad: ', expected_grad, ' actual u grad: ', rnn.u_grad[i, j])
    for i in range(rnn.w.shape[0]):
        for j in range(rnn.w.shape[1]):
            rnn.reset_state()
            rnn.w[i, j] += epsilon
            rnn.forward(x1)
            rnn.forward(x2)
            rnn.forward(x3)
            err1 = error_function(rnn.cell_state_list[-1])
            rnn.reset_state()
            rnn.w[i, j] -= epsilon * 2
            rnn.forward(x1)
            rnn.forward(x2)
            rnn.forward(x3)
            err2 = error_function(rnn.cell_state_list[-1])
            expected_grad = (err1 - err2) / (2 * epsilon)
            rnn.w[i, j] += epsilon
            print('expected w grad: ', expected_grad, ' actual w grad: ', rnn.w_grad[i, j])

if __name__ == '__main__':
    rnn_gradient_check()