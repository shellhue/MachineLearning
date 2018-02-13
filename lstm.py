from fc import SigmoidActivator
import numpy as np


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output


class LstmLayer(object):
    def __init__(self, input_size, output_size, input_activator):
        self.input_size = input_size
        self.output_size = output_size

        self.gate_activator = SigmoidActivator()
        self.output_activator = TanhActivator()
        self.input_activator = input_activator

        self.c_list = self.init_state_vec()
        self.h_list = self.init_state_vec()
        self.f_list = self.init_state_vec()
        self.i_list = self.init_state_vec()
        self.o_list = self.init_state_vec()
        self.ct_list = self.init_state_vec()

        self.wfh, self.wfx, self.bf = self.init_weights_mat()
        self.wih, self.wix, self.bi = self.init_weights_mat()
        self.woh, self.wox, self.bo = self.init_weights_mat()
        self.wch, self.wcx, self.bc = self.init_weights_mat()

        self.times = 0

        self.input_list = []
        self.input_list.append(np.zeros([self.input_size, 1]))

    def init_state_vec(self):
        state_list = [np.zeros([self.output_size, 1])]
        return state_list

    def init_weights_mat(self):
        mat_h = np.random.uniform(-1e-1, 1e-1, [self.output_size, self.output_size])
        mat_x = np.random.uniform(-1e-1, 1e-1, [self.output_size, self.input_size])
        b = np.zeros([self.output_size, 1])
        return mat_h, mat_x, b

    def init_grad_mat(self):
        grad_h = np.zeros([self.output_size, self.output_size])
        grad_x = np.zeros([self.output_size, self.input_size])
        b = np.zeros([self.output_size, 1])
        return grad_h, grad_x, b

    def forward(self, input_array):
        self.input_list.append(input_array)
        self.times += 1
        f_t = self.calc_gate(input_array, self.wfh, self.wfx, self.bf, self.gate_activator)
        i_t = self.calc_gate(input_array, self.wih, self.wix, self.bi, self.gate_activator)
        o_t = self.calc_gate(input_array, self.woh, self.wox, self.bo, self.gate_activator)
        ct_t = self.calc_gate(input_array, self.wch, self.wcx, self.bc, self.output_activator)

        c_t = f_t * self.c_list[-1] + i_t * ct_t

        h = self.output_activator.forward(c_t) * o_t

        self.f_list.append(f_t)
        self.i_list.append(i_t)
        self.o_list.append(o_t)
        self.ct_list.append(ct_t)
        self.c_list.append(c_t)
        self.h_list.append(h)

    def calc_gate(self, x, wh, wx, b, activator):
        previous_output = self.h_list[-1]
        net = np.dot(wh, previous_output) + np.dot(wx, x) + b

        return activator.forward(net)

    def backward(self, delta_h):
        # 计算delta
        delta_f_list = self.init_delta()
        delta_i_list = self.init_delta()
        delta_o_list = self.init_delta()
        delta_ct_list = self.init_delta()

        delta_h_list = self.init_delta()
        delta_h_list[-1] = delta_h

        for t in range(self.times, 0, -1):
            f_t = self.f_list[t]
            i_t = self.i_list[t]
            o_t = self.o_list[t]
            ct_t = self.ct_list[t]
            c_t = self.c_list[t]
            c_t_previous = self.c_list[t - 1]

            delta_h_t = delta_h_list[t]
            tanh_c = self.output_activator.forward(c_t)
            delta_o_t = delta_h_t * tanh_c * self.gate_activator.backward(o_t)
            delta_i_t = delta_h_t * o_t * (1 - tanh_c * tanh_c) * ct_t * self.gate_activator.backward(i_t)
            delta_f_t = delta_h_t * o_t * (1 - tanh_c * tanh_c) * c_t_previous * self.gate_activator.backward(f_t)
            delta_ct_t = delta_h_t * o_t * (1 - tanh_c * tanh_c) * i_t * self.output_activator.backward(ct_t)

            delta_h_t_previous = np.dot(self.woh.T, delta_o_t) + np.dot(self.wfh.T, delta_f_t) + np.dot(self.wih.T, delta_i_t) + np.dot(self.wch.T, delta_ct_t)

            delta_o_list[t] = delta_o_t
            delta_i_list[t] = delta_i_t
            delta_f_list[t] = delta_f_t
            delta_ct_list[t] = delta_ct_t
            delta_h_list[t - 1] = delta_h_t_previous

        # 计算梯度
        self.wfh_grad, self.wfx_grad, self.bf_grad = self.init_grad_mat()
        self.wih_grad, self.wix_grad, self.bi_grad = self.init_grad_mat()
        self.woh_grad, self.wox_grad, self.bo_grad = self.init_grad_mat()
        self.wch_grad, self.wcx_grad, self.bc_grad = self.init_grad_mat()
        for t in range(self.times, 0, -1):
            delta_o_t = delta_o_list[t]
            delta_i_t = delta_i_list[t]
            delta_f_t = delta_f_list[t]
            delta_ct_t = delta_ct_list[t]
            h_t_previous = self.h_list[t - 1]
            input_t = self.input_list[t]
            print(t, np.dot(delta_f_t, h_t_previous.T))

            self.wfh_grad += np.dot(delta_f_t, h_t_previous.T)
            self.wfx_grad += np.dot(delta_f_t, input_t.T)
            self.bf_grad += delta_f_t
            self.wih_grad += np.dot(delta_i_t, h_t_previous.T)
            self.wix_grad += np.dot(delta_i_t, input_t.T)
            self.bi_grad += delta_i_t
            self.woh_grad += np.dot(delta_o_t, h_t_previous.T)
            self.wox_grad += np.dot(delta_o_t, input_t.T)
            self.bo_grad += delta_o_t
            self.wch_grad += np.dot(delta_ct_t, h_t_previous.T)
            self.wcx_grad += np.dot(delta_ct_t, input_t.T)
            self.bc_grad += delta_ct_t
        print(self.wfh_grad)

    def init_delta(self):
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros([self.output_size, 1]))

        return delta_list

    def update(self, learning_rate):
        self.wfh -= learning_rate * self.wfh_grad
        self.wfx -= learning_rate * self.wfx_grad
        self.bf -= learning_rate * self.bf_grad
        self.wih -= learning_rate * self.wih_grad
        self.wix -= learning_rate * self.wix_grad
        self.bi -= learning_rate * self.bi_grad
        self.woh -= learning_rate * self.woh_grad
        self.wox -= learning_rate * self.wox_grad
        self.bo -= learning_rate * self.bo_grad
        self.wch -= learning_rate * self.wch_grad
        self.wcx -= learning_rate * self.wcx_grad
        self.bc -= learning_rate * self.bc_grad

    def reset_state(self):
        self.times = 0
        self.c_list = self.init_state_vec()
        self.h_list = self.init_state_vec()
        self.f_list = self.init_state_vec()
        self.i_list = self.init_state_vec()
        self.o_list = self.init_state_vec()
        self.ct_list = self.init_state_vec()
        self.input_list = []
        self.input_list.append(np.zeros([self.input_size, 1]))


def lstm_gradient_check():

    def error_function(x):
        return x.sum()

    x1 = np.array([[2], [4], [10], [10], [10]])
    x2 = np.array([[8], [6], [5], [10], [10]])
    x3 = np.array([[18], [13], [15], [10], [10]])
    delta_array = np.array([[1], [1], [1]])

    lstm = LstmLayer(5, 3, SigmoidActivator())
    lstm.forward(x1)
    lstm.forward(x2)
    lstm.forward(x3)
    lstm.backward(delta_array)

    epsilon = 0.0001
    for i in range(lstm.wfh.shape[0]):
        for j in range(lstm.wfh.shape[1]):
            lstm.reset_state()
            lstm.wfh[i, j] += epsilon
            lstm.forward(x1)
            lstm.forward(x2)
            lstm.forward(x3)
            err1 = error_function(lstm.h_list[-1])
            lstm.reset_state()
            lstm.wfh[i, j] -= epsilon * 2
            lstm.forward(x1)
            lstm.forward(x2)
            lstm.forward(x3)
            err2 = error_function(lstm.h_list[-1])
            expected_grad = (err1 - err2) / (2 * epsilon)
            lstm.wfh[i, j] += epsilon
            print('expected wfh grad: ', expected_grad, ' actual wfh grad: ', lstm.wfh_grad[i, j])


if __name__ == '__main__':
    lstm_gradient_check()