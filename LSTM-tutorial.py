# following tutorial found at
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

######################## SIMPLE TIME-SERIES ########################

seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_features = 1
n_steps = 3
X, y = split_sequence(seq, 3)
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['MeanSquaredError']
)

model.fit(X, y, epochs=1000)

x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input)
print(yhat)

######################## PARALLEL TIME-SERIES ########################

