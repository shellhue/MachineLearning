from functools import reduce

class Perception(object):
    def __init__(self, input_dimention, activator):
        '''
        感知机初始化

        :param input_dimention: 输入的维数
        :param activator: 激活函数，类型为float -> float
        '''
        self.input_dimention = input_dimention
        self.activator = activator
        self.weights = [0 for _ in range(input_dimention)]
        self.bias = 0

    def train(self, inputs, labels, learning_rate):
        '''
        训练

        :param inputs: 需要训练的数据
        :param labels: 数据标签
        :param learning_rate: 学习率
        :return: 没有返回值
        '''

        predicts = [self.predict(input) for input in inputs]
        return

    def one_iteration(self, ):

    def predict(self, input):
        return self.activator(reduce(lambda a, b: a + b, map(lambda item: item[0] * item[1], zip(input, self.weights)), 0) + self.bias)

def f(x):
    return 1 if x > 0 else 0

if __name__ == '__main__':
    and_perception = Perception(2, f)
