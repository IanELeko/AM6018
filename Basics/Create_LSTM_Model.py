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

# random seeds
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# constants
LSTM_UNITS = 32


#----------------------------------------------------------------------
#######################################################################
# -------------------- GENERATING THE DATASET -------------------------
# init ODE solver with limit cycle centered at (0, 0)
lc = LCSolver.Simple2DLimitCycle(x1c=0, x2c=0)

# creates data of the shape (10000, 2) -> 10000 (x1, x2) data points
if __name__ == '__main__':
    data = lc.solve_nstep(
        x0=np.array([-2,0]),
        t0=0,
        tStep=0.01,
        nStep=10000
    )

    print(data)
    print(data.shape)

#----------------------------------------------------------------------
#######################################################################
# ------------------------ HYPERPARAMETERS ----------------------------

# Optimization parameters:
NUM_TRAINING_ITERATIONS = 100
BATCH_SIZE = 32
SEQ_LENGTH = 50
LEARNING_RATE = 5e-3

# Model parameters: 
RNN_UNITS = 32

# Checkpoint location: 
CHECKPOINT_DIR = '.\\training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "my_ckpt")

#----------------------------------------------------------------------
#######################################################################
# ------------------- MODEL BUILDING AND TRAINING ---------------------

if __name__ == '__main__':

    # init the model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            RNN_UNITS, 
            activation='tanh', 
            #input_shape=(seqlen, 2),
            kernel_initializer=tf.keras.initializers.Identity(),
            recurrent_initializer=tf.keras.initializers.Identity(),
            stateful=True,
            return_sequences=True,
            batch_input_shape=(BATCH_SIZE, None, 2)
        ),
        tf.keras.layers.Dense(2)
    ])

    # init Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # define training function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            yhat = model(x) # -> get prediction of shape (BATCH_SIZE, SEQ_LENGTH, 2)
            loss = tf.keras.losses.mean_squared_error(y, yhat) # calc MSE along rows

        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss
    
    # train
    for iter in tqdm(range(NUM_TRAINING_ITERATIONS)):
        x_batch, y_batch = HelperFunctions.get_batch(data, SEQ_LENGTH, BATCH_SIZE)
        loss = train_step(x_batch, y_batch)

        if iter % 100 == 0:
            model.save_weights(CHECKPOINT_PREFIX)

    # save for use later
    model.save_weights(CHECKPOINT_PREFIX)

    print("\n\n\n ##### TRAINING DONE! \n\n\n")
    