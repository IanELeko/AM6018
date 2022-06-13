import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import tensorflow as tf

dt_fname = r"C:\Users\lekoi\Documents\AM6018\Limit Cycles\Data\x0=[2 2],t=0.01,rk=1000000,seq=250.csv"
ts_fname = r"C:\Users\lekoi\Documents\AM6018\Limit Cycles\Data\TSDataset"

data = limitcycle.LCDataset().load_dataset(dt_fname)
tsdata = limitcycle.LCDataset().load_time_series(ts_fname)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(2, activation='tanh', input_shape=(250, 2)),
    tf.keras.layers.Dense(2)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(tsdata, epochs=10, batch_size=32)

model.save('LSTM2e10_tanh_250')