import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import linearsys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


T_STEP = 0.1
RK_STEPS = 10
LSTM_SEQ_LEN = 5
ls = linearsys.LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))
x0 = np.array([1, 1])
data = ls.solve_nstep(x0=x0, t0=0, tStep=T_STEP, nStep=RK_STEPS)

dataset = keras.utils.timeseries_dataset_from_array(
    data=data[:-1], 
    targets=data[LSTM_SEQ_LEN:], 
    sequence_length=LSTM_SEQ_LEN
)

X, y = list(dataset.take(1).as_numpy_iterator())[0]
Xnorm, ynorm = X.copy(), y.copy()

# normalise the dataset
def normalise(X, y):
    maxes, mins = [], []
    Xnorm, ynorm = X.copy(), y.copy()

    for i in range(X.shape[0]):
        imax = tf.reduce_max(tf.concat((X[i], y), axis=0))
        imin = tf.reduce_min(tf.concat((X[i], y), axis=0))

        Xnorm[i] = (X[i] - imin) / (imax - imin)
        ynorm[i] = (y[i] - imin) / (imax - imin)

        maxes.append(imax)
        mins.append(imin)

    return Xnorm, ynorm, (maxes, mins)

def renormalise_y(ynorm, imax, imin):
    return ynorm * (imax - imin) + imin

print(X)
print(normalise(X, y)[:2])

Xnorm, ynorm, (maxes, mins) = normalise(X, y)

model = keras.Sequential([
    layers.LSTM(2, activation='relu'),
    layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics='mse'
)

model.fit(Xnorm, ynorm)

PRED_LEN = 100
xshat = np.zeros((PRED_LEN, 2))
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


#xsample, ysample = list(dataset.take(1).as_numpy_iterator())[0]

#print(xsample)
#print(ysample)

lstm = layers.LSTM(2, activation='relu')
dense = layers.Dense(2)
bn = layers.BatchNormalization()
ln = layers.LayerNormalization()

#print(lstm(xsample))
#print(dense(lstm(xsample)))
#print(bn(xsample))
#print(ln(xsample))


