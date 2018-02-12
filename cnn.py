import numpy as np


class ConvLayer(object):
    def __init__(self,
                 input_width,
                 input_height,
                 input_channel,
                 filter_width,
                 filter_height,
                 filter_number,
                 zero_padding,
                 stride,
                 input_activator,
                 output_activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filer_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.input_activator = input_activator
        self.output_activator = output_activator
        self.learning_rate = learning_rate

        self.output_width = ConvLayer.calculate_output_size(input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(input_height, filter_height, zero_padding, stride)
        self.output = np.zeros([self.filer_number, self.output_height, self.output_width])

        self.filters = []
        for _ in range(self.filer_number):
            self.filters.append(Filter(self.filter_width, self.filter_height, self.input_channel))

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(self.input_array, self.zero_padding)
        for c in range(self.filer_number):
            f = self.filters[c]
            conv(self.padded_input_array, self.output[c], f.get_weights(), f.get_bias(), self.stride)
        element_wise_op(self.output, self.output_activator)

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size + 2 * zero_padding - filter_size) / stride + 1


def padding(array, zero_padding):
    if array.ndim == 2:
        height = array.shape[0]
        width = array.shape[1]
        padded__array = np.zeros([height + 2 * zero_padding,
                                  width + 2 * zero_padding])
        padded__array[zero_padding: zero_padding + height, zero_padding: zero_padding + width] = array
        return padded__array
    depth = array.shape[0]
    height = array.shape[1]
    width = array.shape[2]
    padded__array = np.zeros([depth,
                              height + 2 * zero_padding,
                              width + 2 * zero_padding])
    padded__array[:, zero_padding: zero_padding + height, zero_padding: zero_padding + width] = array
    return padded__array


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array, output_array, filter_weights, filter_bias, stride):
    for i in range(output_array.shape[0]):
        for j in range(output_array.shape[1]):
            patched_input = get_patch(input_array, i, j, filter_weights.shape[1], filter_weights.shape[2], stride)
            output_array[i, j] = (patched_input * filter_weights).sum() + filter_bias

def get_patch(input_array, i, j, patch_height, patch_width, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[start_i: start_i + patch_height, start_j: start_j + patch_width]
    elif input_array.ndim == 3:
        return input_array[:, start_i: start_i + patch_height, start_j: start_j + patch_width]


class Filter(object):
    def __init__(self, width, height, depth):
        self.w = np.random.uniform(-1e-4, 1e-4, [depth, height, width])
        self.b = 0
        self.w_grad = np.zeros(self.w.shape)
        self.b_grad = 0

    def update(self, learning_rate):
        self.w -= learning_rate * self.w_grad
        self.b -= learning_rate * self.b_grad

    def get_weights(self):
        return self.w

    def get_bias(self):
        return self.b


if __name__ == '__main__':
    a = np.random.uniform(-1e-4, 1e-4, [3, 1])
    b = np.zeros([1, 5])
    v = np.array([[0, 2], [2, 3]])
    v = padding(v, 2)
    g = np.array([[1, 2], [3, 4]])
    print(element_wise_op(v, lambda x: 2 * x))
    print(v)
