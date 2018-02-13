from fc import SigmoidActivator
import numpy as np
from functools import reduce


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

        self.wfh, self.wfx, self.wfb = self.init_weights_mat()
        self.wih, self.wix, self.wib = self.init_weights_mat()
        self.woh, self.wox, self.wob = self.init_weights_mat()
        self.wch, self.wcx, self.wcb = self.init_weights_mat()

        self.wfh_grad, self.wfx_grad, self.wfb_grad = self.init_grad_mat()
        self.wih_grad, self.wix_grad, self.wib_grad = self.init_grad_mat()
        self.woh_grad, self.wox_grad, self.wob_grad = self.init_grad_mat()
        self.wch_grad, self.wcx_grad, self.wcb_grad = self.init_grad_mat()

        self.times = 0

        self.input_list = []

    def init_state_vec(self):
        state_list = [np.zeros([self.output_size, 1])]
        return state_list

    def init_weights_mat(self):
        mat_h = np.random.uniform(-1e-4, 1e-4, [self.output_size, self.output_size])
        mat_x = np.random.uniform(-1e-4, 1e-4, [self.output_size, self.input_size])
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
        f_t = self.calc_gate(input_array, self.wfh, self.wfx, self.wfb, self.gate_activator)
        i_t = self.calc_gate(input_array, self.wih, self.wix, self.wib, self.gate_activator)
        o_t = self.calc_gate(input_array, self.woh, self.wox, self.wob, self.gate_activator)
        ct_t = self.calc_gate(input_array, self.wch, self.wcx, self.wcb, self.output_activator)

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
            delta_ct_t = delta_h_t * o_t * (1 - tanh_c * tanh_c) * i_t * (1 - ct_t * ct_t)

            delta_h_t_previous = np.dot(self.woh.T, delta_o_t) + np.dot(self.wfh, delta_f_t) + \
                                 np.dot(self.wih, delta_i_t) + np.dot(self.wch, delta_ct_t)

            delta_o_list[t] = delta_o_t
            delta_i_list[t] = delta_i_t
            delta_f_list[t] = delta_f_t
            delta_ct_list[t] = delta_ct_t
            delta_h_list[t - 1] = delta_h_t_previous

        # 计算梯度
        wfh_grad_list = []
        wfx_grad_list = []
        wfb_grad_list = []
        wih_grad_list = []
        wix_grad_list = []
        wib_grad_list = []
        woh_grad_list = []
        wox_grad_list = []
        wob_grad_list = []
        wch_grad_list = []
        wcx_grad_list = []
        wcb_grad_list = []

        for t in range(self.times, 0, -1):
            delta_o_t = delta_o_list[t]
            delta_i_t = delta_i_list[t]
            delta_f_t = delta_f_list[t]
            delta_ct_t = delta_ct_list[t]
            h_t_previous = self.h_list[t - 1]
            input_t = self.input_list[t]

            wfh_grad_list.append(np.dot(delta_f_t, h_t_previous.T))
            wfx_grad_list.append(np.dot(delta_f_t, input_t.T))
            wfb_grad_list.append(delta_f_t)
            wih_grad_list.append(np.dot(delta_i_t, h_t_previous.T))
            wix_grad_list.append(np.dot(delta_i_t, input_t.T))
            wib_grad_list.append(delta_i_t)
            woh_grad_list.append(np.dot(delta_o_t, h_t_previous.T))
            wox_grad_list.append(np.dot(delta_o_t, input_t.T))
            wob_grad_list.append(delta_o_t)
            wch_grad_list.append(np.dot(delta_ct_t, h_t_previous.T))
            wcx_grad_list.append(np.dot(delta_ct_t, input_t.T))
            wcb_grad_list.append(delta_ct_t)

        self.wfh_grad = reduce(lambda m, n: n + m, wfh_grad_list, np.zeros(self.wfh_grad.shape))
        self.wfx_grad = reduce(lambda m, n: n + m, wfx_grad_list, np.zeros(self.wfx_grad.shape))
        self.wfb_grad = reduce(lambda m, n: n + m, wfb_grad_list, np.zeros(self.wfb_grad.shape))
        self.wih_grad = reduce(lambda m, n: n + m, wih_grad_list, np.zeros(self.wih_grad.shape))
        self.wix_grad = reduce(lambda m, n: n + m, wix_grad_list, np.zeros(self.wix_grad.shape))
        self.wib_grad = reduce(lambda m, n: n + m, wib_grad_list, np.zeros(self.wib_grad.shape))
        self.woh_grad = reduce(lambda m, n: n + m, woh_grad_list, np.zeros(self.woh_grad.shape))
        self.wox_grad = reduce(lambda m, n: n + m, wox_grad_list, np.zeros(self.wox_grad.shape))
        self.wob_grad = reduce(lambda m, n: n + m, wob_grad_list, np.zeros(self.wob_grad.shape))
        self.wch_grad = reduce(lambda m, n: n + m, wch_grad_list, np.zeros(self.wch_grad.shape))
        self.wcx_grad = reduce(lambda m, n: n + m, wcx_grad_list, np.zeros(self.wcx_grad.shape))
        self.wcb_grad = reduce(lambda m, n: n + m, wcb_grad_list, np.zeros(self.wcb_grad.shape))

    def init_delta(self):
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros([self.output_size, 1]))

        return delta_list

    def update(self, learning_rate):
        self.wfh -= learning_rate * self.wfh_grad
        self.wfx -= learning_rate * self.wfx_grad
        self.wfb -= learning_rate * self.wfb_grad
        self.wih -= learning_rate * self.wih_grad
        self.wix -= learning_rate * self.wix_grad
        self.wib -= learning_rate * self.wib_grad
        self.woh -= learning_rate * self.woh_grad
        self.wox -= learning_rate * self.wox_grad
        self.wob -= learning_rate * self.wob_grad
        self.wch -= learning_rate * self.wch_grad
        self.wcx -= learning_rate * self.wcx_grad
        self.wcb -= learning_rate * self.wcb_grad


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 5], [1, 2, 3, 8]])
    np.insert(a, 0, [1, 2, 3, 5])
    print(a)