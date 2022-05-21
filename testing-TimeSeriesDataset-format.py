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
print('Data')
print(data)
print('-----------------')

X = data[:-1]
print(X)
print('-----------------')
y = data[LSTM_SEQ_LEN:]
print(y)
print('-----------------')

dataset = keras.utils.timeseries_dataset_from_array(
    data=X, 
    targets=y, 
    sequence_length=LSTM_SEQ_LEN
)

print('Elements')
for element in dataset:
    print(element)