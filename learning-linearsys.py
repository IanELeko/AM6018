import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import linearsys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

ls = linearsys.LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))
x0 = np.array([1, 1])
xs = ls.solve(x0, 0, 100, 0.1)

N_STEP = 5
N_FEAT = 2
X, y = split_sequence(xs, N_STEP)

x_init = X[0]



model = keras.Sequential([
    layers.LSTM(1024, activation='relu', input_shape=(N_STEP, N_FEAT)),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X, y, epochs=10)

plt.plot(xs[:, 0], xs[:, 1], 'r--')

xshat = np.zeros((1001, N_FEAT))
xshat[:5, :] = x_init
for i in range(N_STEP, xshat.shape[0]):
    if i % 100 == 0:
        print(f'Predicting: {i}')
    xshat[i] = model.predict(xshat[i-5:i, :].reshape(1, N_STEP, N_FEAT), verbose=0)

plt.plot(xshat[:, 0], xshat[:, 1], 'b')
plt.show()




