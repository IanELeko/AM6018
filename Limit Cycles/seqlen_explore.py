import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import tensorflow as tf

seqlens = [5 * i for i in range(1, 11)]

dt_fname = r"C:\Users\lekoi\Documents\AM6018\Limit Cycles\Data\x0=[2 2],t=0.01,rk=10000,seq=250.csv"
data = limitcycle.LCDataset().load_dataset(dt_fname)
print(len(data))

prederrs = np.zeros(len(seqlens))
k = 0
for seqlen in seqlens:
    print(f"\n\nSEQUENCE LENGTH: {seqlen}\n\n")
    ts_data = tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1], 
            targets=data[seqlen:], 
            sequence_length=seqlen
        )

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, activation='tanh', input_shape=(seqlen, 2)),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    model.fit(ts_data, epochs=10, batch_size=32)

    SEQ_LEN = seqlen
    PRED_LEN = 1000
    x_init = data[:SEQ_LEN]
    xshat = np.zeros((PRED_LEN, 2))
    xshat[:SEQ_LEN, :] = x_init

    diffs = np.zeros(PRED_LEN)

    for i in range(SEQ_LEN, PRED_LEN):
        if i % 100 == 0:
            print(f'Prediction number: {i}')
        xshat_point = xshat[i-SEQ_LEN:i, :]
        xshat_point = xshat_point.reshape(1, SEQ_LEN, 2)
        yhat = model.predict(xshat_point, verbose=0)
        diffs[i] = np.sum((data[i] - yhat)**2)
        xshat[i, :] = yhat

    prederrs[k] = np.sum(diffs)
    k += 1

print(prederrs)
