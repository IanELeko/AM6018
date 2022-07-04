import numpy as np
import tensorflow as tf

y = np.array([[1, 2], [3, 4]], dtype=np.float32)
yhat = np.array([[0, 5], [1, 3]],dtype=np.float32)

print(tf.keras.losses.mean_squared_error(y, yhat))