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
        # self.filters = []
        # for _ in range(input_channel):

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size + 2 * zero_padding - filter_size) / stride + 1




if __name__ == '__main__':
    print('hahaah')
    print(np)