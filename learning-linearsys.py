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

T_LEN = 1000
ls = linearsys.LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))
x0 = np.array([1, 1])
xs = ls.solve_nstep(x0=x0, t0=0, tStep=0.1, nStep=T_LEN)

N_STEP = 5
N_FEAT = 2
X, y = split_sequence(xs, N_STEP)

x_init = X[0]



model = keras.Sequential([
    layers.BatchNormalization(input_shape=(N_STEP, N_FEAT)),
    layers.LSTM(1024, activation='relu'),
    layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X, y, epochs=1, batch_size=32)

plt.plot(xs[:, 0], xs[:, 1], 'r--')

PRED_LEN = 100
xshat = np.zeros((PRED_LEN, N_FEAT))
xshat[:N_STEP, :] = x_init

'''for i in range(200):
    try:
        print(xshat[i])
    except:
        break'''

for i in range(N_STEP, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    pred = model.predict(xshat[i-N_STEP:i, :].reshape(1, N_STEP, N_FEAT), verbose=0)
    print(f'Prediction: {pred}')
    xshat[i, :] = pred

plt.plot(xshat[:, 0], xshat[:, 1], 'b')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

np.savetxt('xshat.csv', xshat)

'''for i in range(10000):
    try:
        print(xshat[i])
    except:
        break'''



