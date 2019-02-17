import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense
from keras.utils import np_utils
from keras.layers.core import Dense, Activation


def convertToBinary(int):
   bin_list = []
   for i in range(8):
       bin_list.append(int % 2)
       int = int // 2

   return bin_list[::-1]


# X = np.array(range(255), dtype=int)
# y = np.array(list(map(lambda x: convertToBinary(x), X)))

length = 60
X = np.array([i * 2 * np.pi / length for i in range(length)])
y = np.sin(X)
# X = np.array([i / 20 for i in range(20)])
# y = X * 2

X = (np.random.random((100000))-.5) * 2 * np.pi
y = np.sin(X)

plt.plot(X, y, '.')

model = Sequential()
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()
model.fit(X, y, batch_size=32, nb_epoch=5, verbose=1)

X_test = X + 0.3
y_pred = np.squeeze(model.predict(X_test))
plt.plot(X_test, y_pred, '*')
y_pred = np.squeeze(model.predict(X))
plt.plot(X, y_pred, '*')

###aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
############################################################################
############################################################################
############################################################################
############################################################################

# model = Sequential()
# model.add(Dense(output_dim=5, input_dim=1))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=5, input_dim=1))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=5, input_dim=1))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=5, input_dim=1))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=1))
#
# model.compile(loss='mean_squared_error', optimizer='sgd')
#
#
# x = np.linspace(-np.pi, np.pi, 100)
#
# data = (np.random.random((100000))-.5) * 2 * np.pi
# vals = np.sin(data)
#
# model.fit(data, vals, nb_epoch=5, batch_size=32)
#
# y = model.predict(x, batch_size=32, verbose=0)
# plt.plot(x,y)
#
# plt.plot(x,np.sin(x))
# #plt.legend()
# plt.show()



