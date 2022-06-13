import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

##### set-up variables #####

x0 = np.array([2, 2])
T_STEP = 0.1
RK_STEPS = 10000
SEQ_LEN = 250

##### generate data #####

lcdata = limitcycle.SimpleLimitCycleDataGenerator(
    x0=x0,
    t_step=T_STEP,
    rk_steps=RK_STEPS,
    seq_len=SEQ_LEN
)

data = lcdata.generate_data()
ts_data = lcdata.generate_time_series()

model = keras.Sequential([
    layers.LSTM(32, activation='tanh', input_shape=(SEQ_LEN, 2)),
    layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(ts_data, epochs=10, batch_size=32)

PRED_LEN = 1000
x_init = data[:SEQ_LEN]
xshat = np.zeros((PRED_LEN, 2))
xshat[:SEQ_LEN, :] = x_init


for i in range(SEQ_LEN, PRED_LEN):
    if i % 100 == 0:
        print(f'Prediction number: {i}')
    xshat_point = xshat[i-SEQ_LEN:i, :]
    xshat_point = xshat_point.reshape(1, SEQ_LEN, 2)
    yhat = model.predict(xshat_point, verbose=0)
    xshat[i, :] = yhat

plt.plot(data[:-1, 0], data[:-1, 1], 'r--')
plt.plot(xshat[:, 0], xshat[:, 1], 'b')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()