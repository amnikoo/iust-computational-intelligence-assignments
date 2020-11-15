from __future__ import print_function

# *-* coding: utf-8 *-*

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout
from keras.optimizers import RMSprop, SGD

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


epochs = 80

inp = np.linspace(-2,2,41)
out = inp*inp*inp

model = Sequential()
model.add(Dense(41, input_dim=1, activation='tanh'))
model.add(RBFLayer(10, 0.5))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = model.fit(inp, out,
                    epochs=epochs,
                    verbose=1)

res = model.predict(inp)
r = plt.figure(1)
plt.plot(inp, res, 'r', label="learned")
plt.plot(inp, out, 'b', label="x^3")
plt.legend()
plt.savefig('RBF.png')
r.show()
