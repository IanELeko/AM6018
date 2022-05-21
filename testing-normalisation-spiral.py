import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import linearsys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def normalise_one(x):
    xnorm = x.copy()

    for i in range(x.shape[0]):
        imax = tf.reduce_max(x, axis=0)
        imin = tf.reduce_min(x, axis=0)

        xnorm[i] = (x[i] - imin) / (imax - imin)

    return xnorm, (imax, imin)


def normalise(X, y):
    maxes, mins = [], []
    Xnorm, ynorm = X, y

    for i in range(X.shape[0]):
        imax = tf.reduce_max(tf.concat((X[i], y), axis=0))
        imin = tf.reduce_min(tf.concat((X[i], y), axis=0))

        Xnorm[i] = (X[i] - imin) / (imax - imin)
        ynorm[i] = (y[i] - imin) / (imax - imin)

        maxes.append(imax)
        mins.append(imin)

    return Xnorm, ynorm, (maxes, mins)

T_STEP = 0.1
RK_STEPS = 10000
LSTM_SEQ_LEN = 25
ls = linearsys.LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))
x0 = np.array([1, 1])
data = ls.solve_nstep(x0=x0, t0=0, tStep=T_STEP, nStep=RK_STEPS)
print(data.shape)

dataset = keras.utils.timeseries_dataset_from_array(
    data=data[:-1], 
    targets=data[LSTM_SEQ_LEN:], 
    sequence_length=LSTM_SEQ_LEN
)

dataset_norm = dataset.map(normalise)

print(f'Cardinality: {dataset.cardinality().numpy()}')
for element in dataset:
    print('Hey, an element!')

X, y = list(dataset.take(1).as_numpy_iterator())[0]
print(X.shape, y.shape)
Xnorm, ynorm, (maxes, mins) = normalise(X, y)
print(Xnorm.shape, ynorm.shape)
x_init = data[:LSTM_SEQ_LEN]

model = keras.Sequential([
    layers.LSTM(1024, activation='relu', input_shape=(LSTM_SEQ_LEN, 2)),
    layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(Xnorm, ynorm, epochs=10, batch_size=32)

PRED_LEN = 100
xshat = np.zeros((PRED_LEN, 2))
xshat[:LSTM_SEQ_LEN, :] = x_init


for i in range(LSTM_SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xshat_norm, (imax, imin) = normalise_one(xshat[i-LSTM_SEQ_LEN:i, :])
    xshat_norm = xshat_norm.reshape(1, LSTM_SEQ_LEN, 2)
    yhat_norm = model.predict(xshat_norm, verbose=0)
    yhat = yhat_norm * (imax - imin) + imin
    print(f'Prediction: {yhat}')
    xshat[i, :] = yhat

plt.plot(data[:, 0], data[:, 1], 'r--')
plt.plot(xshat[:, 0], xshat[:, 1], 'b')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()