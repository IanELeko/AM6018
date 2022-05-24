##### imports #####

import sys
sys.path.append("C:\\Users\\lekoi\\Documents\\AM6018\\")

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import linearsys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

##### get training data #####

T_STEP = 0.01
RK_STEPS = 10000
LSTM_SEQ_LEN = 10

ls = linearsys.LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))
x0 = np.array([1, 1])
data = ls.solve_nstep(x0=x0, t0=0, tStep=T_STEP, nStep=RK_STEPS)

##### split and normalise the data #####

def normalise_minmax(x):
    xnorm = x

    for i in range(x.shape[0]):
        imax = tf.reduce_max(x, axis=0)
        imin = tf.reduce_min(x, axis=0)

        xnorm = (x - imin) / (imax - imin)

    return xnorm, (imin, imax)

def normalise_z(x):
    xnorm = x

    for i in range(x.shape[0]):
        avg = tf.reduce_mean(x, axis=0)
        std = tf.math.reduce_std(x, axis=0)

        xnorm = (x - avg) / std

    return xnorm

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break

        #print(f'Sequence: {sequence[i:end_ix+1, :]}')
        seq_norm = normalise_minmax(sequence[i:end_ix+1, :])[0]
        #print(f'seq_norm: {seq_norm}')
        seq_x, seq_y = seq_norm[:-1, :], seq_norm[-1, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

print('Splitting the sequence...')
X, y = split_sequence(data, LSTM_SEQ_LEN)

shuffler = np.random.permutation(X.shape[0])
X = X[shuffler]
y = y[shuffler]


print('Splitting done...')
'''print('X ----------------------- X')
print(X)
print('y ----------------------- y')
print(y)'''

##### model #####

model = keras.Sequential([
    layers.LSTM(1024, activation='relu', input_shape=(LSTM_SEQ_LEN, 2)),
    layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(X, y, epochs=5, batch_size=32)

##### predict and plot #####

PRED_LEN = 1000

x_init = data[:LSTM_SEQ_LEN]
xshat = np.zeros((PRED_LEN, 2))
xshat[:LSTM_SEQ_LEN, :] = x_init
for i in range(LSTM_SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xpred = xshat[i-LSTM_SEQ_LEN:i, :]
    xpred_norm, (imin, imax) = normalise_minmax(xpred)
    #pred = model.predict(xpred_norm.reshape(1, LSTM_SEQ_LEN, 2), verbose=0)
    pred = model.predict(tf.reshape(xpred_norm, (1, LSTM_SEQ_LEN, 2)), verbose=0)
    #print(f'Prediction: {pred}')
    yhat = pred * (imax - imin) + imin
    xshat[i, :] = yhat

plt.plot(data[:, 0], data[:, 1], 'r--')
plt.plot(xshat[:, 0], xshat[:, 1], 'b')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()