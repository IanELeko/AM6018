import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import linearsys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data = np.array([n for n in range(0, 100, 10)], dtype=np.float32)
print(data)

dataset = keras.utils.timeseries_dataset_from_array(
    data=data[:-1], 
    targets=data[5:], 
    sequence_length=5,
    batch_size=1
)

print('Elements!')
for element in dataset:
    print(element)
    #print(element[0].dtype, element[1].dtype)

def normalise(x, y):
    #x = tf.constant(x, dtype=tf.float32)
    #print(x.shape)
    y = tf.expand_dims(y, axis=0)
    data=tf.concat((x, y), axis=1)

    imax = tf.math.reduce_max(data)
    imin = tf.math.reduce_min(data)

    return (x - imin)/(imax - imin), (y - imin)/(imax - imin)


'''
[x, y] = list(dataset.take(1).as_numpy_iterator())[0]
print(x)
print(y)

x = tf.constant(x, dtype=tf.float32)
y = tf.expand_dims(tf.constant(y, dtype=tf.float32), axis=0)

print(tf.concat((x, y), axis=1))
'''

newdataset = dataset.map(normalise)

print('New elements!')
for element in newdataset:
    print(element)