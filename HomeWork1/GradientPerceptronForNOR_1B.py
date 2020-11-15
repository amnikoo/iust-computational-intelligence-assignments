import numpy as np
from matplotlib import pyplot as plt
import math


class Perceptron:
    def __init__(self, train_data, lr=0.2, epoch=10):
        self.train_data = train_data  # type: np.array
        self.learn_rate = lr  # type: float
        self.epoch = epoch  # type: int
        self.weights = np.random.rand(len(train_data[0]) - 1)
        self.error_arr = []

    def predict(self, data):
        """
        :rtype: int
        :type data: np.array
        """
        activation = np.dot(data, self.weights)
        return 1 if activation >= -1 else 0

    def train_weight(self):
        for i in range(self.epoch):
            for row in self.train_data:
                error = row[-1] - self.predict(row[:-1])
                self.error_arr.append(error ** 2)
                self.weights += self.learn_rate * error * row[:-1]

        return self.weights


data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
p = Perceptron(data, 0.2, 10)
w = p.train_weight()

plt.scatter([1, 0, 1], [0, 1, 1])
plt.scatter(0, 0, color='red')
x1, y1 = [0, -1 / w[1]], [-1 / w[0], 0]
plt.plot(x1, y1, marker='o')
print('weights : ', w)

plt.show()
