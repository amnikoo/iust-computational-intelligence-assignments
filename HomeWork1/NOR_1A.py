import numpy as np


class NOR:
    def __init__(self, x1, x2):
        self.X = np.array([x1, x2])
        self.weights = np.array([-2, -2])

    def predict(self):
        sum = np.sum(np.multiply(self.X, self.weights))
        return 1 if sum >= -1 else 0


input = [(0, 0), (1, 0), (0, 1), (1, 1)]
print("NOR Perceptron :")
for x in input:
    y = NOR(x[0], x[1])
    print(str(x), " : ", str(y.predict()))

