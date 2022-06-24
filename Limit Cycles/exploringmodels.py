import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import tensorflow as tf

dt_fname = r"C:\Users\lekoi\Documents\AM6018\Limit Cycles\Data\x0=[2 2],t=0.01,rk=10000,seq=250.csv"
ts_fname = r"C:\Users\lekoi\Documents\AM6018\Limit Cycles\Data\TSDataset"

data = limitcycle.LCDataset().load_dataset(dt_fname)

model = tf.keras.models.load_model('LSTM2e10_tanh_250')

SEQ_LEN = 250
PRED_LEN = 10000
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
