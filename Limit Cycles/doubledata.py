import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTHONHASHSEED"] = "0"
import tensorflow as tf

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


lc1 = limitcycle.SimpleLimitCycle(x1c=-0.5, x2c=0)
lc2 = limitcycle.SimpleLimitCycle(x1c=+0.5, x2c=0)
lc4 = limitcycle.SimpleLimitCycle(x1c=0, x2c=0)

data1 = lc1.solve_nstep(
    x0=np.array([-1,0]),
    t0=0,
    tStep=0.1,
    nStep=9999
)

data2 = lc2.solve_nstep(
    x0=np.array([1,0]),
    t0=0,
    tStep=0.1,
    nStep=9999
)

data4 = lc4.solve_nstep(
    x0=np.array([4,4]),
    t0=0,
    tStep=0.1,
    nStep=9999
)



#plt.plot(data1[:-1, 0], data1[:-1, 1], 'r--')
#plt.plot(data2[:-1, 0], data2[:-1, 1], 'b--')
#plt.xlim(-4, 4)
#plt.ylim(-4, 4)
#plt.show()

#data = np.concatenate((data1, data2), axis=0)

#data = tf.random.shuffle(tf.constant(data))

seqlen = 50
tsdata1 = tf.keras.utils.timeseries_dataset_from_array(
    data=data1[:-1], 
    targets=data1[seqlen:], 
    sequence_length=seqlen
)
tsdata2 = tf.keras.utils.timeseries_dataset_from_array(
    data=data2[:-1], 
    targets=data2[seqlen:], 
    sequence_length=seqlen
)

tsdata = tsdata1.concatenate(tsdata2).shuffle(buffer_size=100000)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        32, 
        activation='tanh', 
        input_shape=(seqlen, 2),
        kernel_initializer=tf.keras.initializers.Identity(),
        recurrent_initializer=tf.keras.initializers.Identity()
    ),
    tf.keras.layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(tsdata, epochs=10, batch_size=32)

SEQ_LEN = seqlen
PRED_LEN = 1000

x1_init = data1[:SEQ_LEN]
xshat1 = np.zeros((PRED_LEN, 2))
xshat1[:SEQ_LEN, :] = x1_init
for i in range(SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xshat1_point = xshat1[i-SEQ_LEN:i, :]
    xshat1_point = xshat1_point.reshape(1, SEQ_LEN, 2)
    yhat1 = model.predict(xshat1_point, verbose=0)
    xshat1[i, :] = yhat1

x2_init = data2[:SEQ_LEN]
xshat2 = np.zeros((PRED_LEN, 2))
xshat2[:SEQ_LEN, :] = x2_init
for i in range(SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xshat2_point = xshat2[i-SEQ_LEN:i, :]
    xshat2_point = xshat2_point.reshape(1, SEQ_LEN, 2)
    yhat2 = model.predict(xshat2_point, verbose=0)
    xshat2[i, :] = yhat2

x4_init = data4[:SEQ_LEN]
xshat4 = np.zeros((PRED_LEN, 2))
xshat4[:SEQ_LEN, :] = x4_init
for i in range(SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xshat4_point = xshat4[i-SEQ_LEN:i, :]
    xshat4_point = xshat4_point.reshape(1, SEQ_LEN, 2)
    yhat4 = model.predict(xshat4_point, verbose=0)
    xshat4[i, :] = yhat4

plt.plot(data1[:, 0], data1[:, 1], 'y')
plt.plot(data2[:, 0], data2[:, 1], 'k')
plt.plot(xshat1[:, 0], xshat1[:, 1], 'r--')
plt.plot(xshat2[:, 0], xshat2[:, 1], 'b--')
plt.plot(xshat4[:, 0], xshat4[:, 1], 'g--')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()