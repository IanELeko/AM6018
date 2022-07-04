#######################################################################
# ------------------------- IMPORT AND INIT ---------------------------

import numpy as np
import matplotlib.pyplot as plt
import LCSolver
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0" # turn off GPU for reproducability
os.environ["PYTHONHASHSEED"] = "0" # random seed
import tensorflow as tf
from tqdm import tqdm
import HelperFunctions
import Create_LSTM_Model

# random seeds
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# constants
LSTM_UNITS = 32


#----------------------------------------------------------------------
#######################################################################
# ----------------------- CREATE THE MODEL ----------------------------

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        Create_LSTM_Model.RNN_UNITS, 
        activation='tanh', 
        kernel_initializer=tf.keras.initializers.Identity(),
        recurrent_initializer=tf.keras.initializers.Identity(),
        stateful=True,
        return_sequences=True,
        batch_input_shape=(1, None, 2)
    ),
    tf.keras.layers.Dense(2)
])

print(Create_LSTM_Model.CHECKPOINT_PREFIX)
model.load_weights(tf.train.latest_checkpoint(Create_LSTM_Model.CHECKPOINT_DIR))

#model.build(tf.TensorShape([1, None]))

def generate_sequence(model, start_sequence, generation_length=10000):

    nstart = start_sequence.shape[0]

    x = tf.expand_dims(start_sequence, 0) # add a "batch_size" dimension to the start_sequence

    # empty array to store our results
    seq_generated = np.zeros((start_sequence.shape[0] + generation_length, 2))
    seq_generated[:start_sequence.shape[0]] = start_sequence

    model.reset_states()
    tqdm._instances.clear()

    #print(model(np.array([[[-1, -1]]])))
    #model.reset_states()

    for i in tqdm(range(nstart, nstart+generation_length)):
        yhat = model(x)
        yhat = tf.squeeze(yhat, 0)

        seq_generated[i] = yhat

        x = tf.expand_dims(yhat, 0)

    return seq_generated

# generate data

lc = LCSolver.Simple2DLimitCycle(x1c=0, x2c=0)
data = lc.solve_nstep(
        x0=np.array([-2,0]),
        t0=0,
        tStep=0.01,
        nStep=10000
    )
newseq = generate_sequence(model, start_sequence=np.array([[-1., -1.]]))
plt.plot(newseq[:, 0], newseq[:, 1], 'r', label="LSTM")
plt.plot(data[:, 0], data[:, 1], 'k--', label="RK45 training data")
plt.legend(loc="upper right")
plt.show()
print(newseq)