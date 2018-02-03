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

    def __str__(self):
        return 'weights\t:%s\nbias\t%f' % (self.weights, self.bias)

    def train(self, input_vecs, labels, iterations, learning_rate):
        '''
        训练

        :param input_vecs: 需要训练的数据
        :param labels: 数据标签
        :param iterations: 循环次数
        :param learning_rate: 学习率
        :return: 没有返回值
        '''
        for _ in range(iterations):
            self._one_iteration(input_vecs, labels, learning_rate)

    def _one_iteration(self, input_vecs, labels, learning_rate):
        for (input_vec, label) in zip(input_vecs, labels):
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, learning_rate)

    def _update_weights(self, input_vec, output, label, learning_rate):
        delta = label - output
        self.weights = list(map(lambda item: item[1] + delta * item[0] * learning_rate, list(zip(input_vec, self.weights))))
        self.bias += delta * learning_rate

    def predict(self, input):
        return self.activator(reduce(lambda a, b: a + b, map(lambda item: item[0] * item[1], list(zip(input, self.weights))), 0) + self.bias)

def f(x):
    return 1 if x > 0 else 0

def get_and_training_dataset():
    data = [[0, 0], [0, 1], [1, 1], [1, 0]]
    labels = [0, 0, 1, 0]
    return data, labels

def get_or_training_dataset():
    data = [[0, 0], [0, 1], [1, 1], [1, 0]]
    labels = [0, 1, 1, 1]
    return data, labels

def train_or_perception():
    or_perception = Perception(2, f)
    input_vecs, labels = get_or_training_dataset()
    or_perception.train(input_vecs, labels, 100, 0.1)

    return or_perception

def train_and_perception():
    and_perception = Perception(2, f)
    input_vects, labels = get_and_training_dataset()
    and_perception.train(input_vects, labels, 100, 0.1)

    return and_perception

if __name__ == '__main__':
    or_perception = train_or_perception()
    print(or_perception)
    print(or_perception.predict([0, 1]))
    print(or_perception.predict([1, 1]))
    print(or_perception.predict([0, 0]))
    print(or_perception.predict([1, 0]))

