import numpy as np
import matplotlib.pyplot as plt
import math


class Perceptron:
    def __init__(self, train_data, lr=0.2, epoch=10):
        self.train_data = train_data  # type: np.array
        self.learn_rate = lr  # type: float
        self.epoch = epoch  # type: int
        self.weights = np.random.rand(len(train_data[0]) - 1)
        self.error_arr = []
        self.bias = 1

    def predict(self, data):
        """
        :rtype: int
        :type data: np.array
        """
        activation = np.dot(data, self.weights)
        return 1 if activation >= 1 else 0

    def train_weight(self):
        for i in range(self.epoch):
            total_err = 0
            for row in self.train_data:
                error = row[-1] - self.predict(row[:-1])
                total_err += error ** 2
                self.bias += self.learn_rate * error
                self.weights += self.learn_rate * error * row[:-1]

            self.error_arr.append(total_err)

        return self.weights, self.bias


def get_dataset():
    dataset = []
    with open('data.txt') as f:
        for line in f:
            row = line[:-1].split(',')
            row[0] = float(format(float(row[0]) / 100, '.9f'))
            row[1] = float(format(float(row[1]) / 100, '.9f'))
            row[2] = int(float(row[2]))
            dataset.append(row)

    return dataset


dataset = np.array(get_dataset())

p = Perceptron(dataset, 0.0001, 3000)
weights, bias = p.train_weight()

x0 = dataset[dataset[:, 2] == 0][:, 0]
y0 = dataset[dataset[:, 2] == 0][:, 1]

x1 = dataset[dataset[:, 2] == 1][:, 0]
y1 = dataset[dataset[:, 2] == 1][:, 1]

plt.scatter(x0, y0, color=['red'])
plt.scatter(x1, y1, color=['blue'])
w0 = -1 / weights[0]
w1 = -1 / weights[1]

x1, y1 = [0, w1], [w0, 0]
plt.plot(x1, y1, marker='o')
plt.figure()
plt.plot(p.error_arr)
print("bias: {0}, w: {1}".format(p.bias, p.weights))

plt.show()
