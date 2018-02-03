from perception import  Perception

f = lambda x: x

class LinearUnit(Perception):
    def __init__(self, input_num):
        Perception.__init__(self, input_num, f)

def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    linear_unit = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    linear_unit.train(input_vecs, labels, 10, 0.01)
    return linear_unit
if __name__ == '__main__':
    print('=====')
    linear_unit = train_linear_unit()
    print(linear_unit)
    print(linear_unit.predict([3.4]))
    print(linear_unit.predict([15]))
    print(linear_unit.predict([1.5]))
    print(linear_unit.predict([6.3]))