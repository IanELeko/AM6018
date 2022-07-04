import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTHONHASHSEED"] = "0"
import tensorflow as tf
from tqdm import tqdm

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


# -------------------- GENERATING THE DATASET -------------------------
lc = limitcycle.SimpleLimitCycle(x1c=0, x2c=0)
data = lc.solve_nstep(
    x0=np.array([-1,0]),
    t0=0,
    tStep=0.1,
    nStep=9999
)

seqlen = 50

def get_batch(dataset, sequence_length, batch_size):
    # length of the dataset
    n = dataset.shape[0] - 1
    
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - sequence_length, batch_size) # list of lenght batch_size

    # get batches
    input_batch = [dataset[ind:ind+sequence_length] for ind in idx]
    output_batch = [dataset[ind+1:ind+sequence_length+1] for ind in idx] # list of numpy arrays of shape (5, 2)

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, sequence_length, 2])
    y_batch = np.reshape(output_batch, [batch_size, sequence_length, 2])
    return x_batch, y_batch

# test get_batch
# print(get_batch(data, 5, 2))

#-------------------------------------------------------------------

# --------------------- MODEL BUILDING -----------------------------
BATCH_SIZE = 32

def get_model(num_units, init_diag=1, batch_size=1):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            num_units, 
            activation='tanh', 
            #input_shape=(seqlen, 2),
            kernel_initializer=tf.keras.initializers.Identity(init_diag),
            recurrent_initializer=tf.keras.initializers.Identity(init_diag),
            stateful=True,
            return_sequences=True,
            batch_input_shape=(batch_size, None, 2)
        ),
        tf.keras.layers.Dense(2)
    ])
    return model

model = get_model(32, batch_size=BATCH_SIZE)
print(model.summary())

# checking model dimensions
xcheck, ycheck = get_batch(data, 50, BATCH_SIZE)
pred = model(xcheck)
print("Input shape:      ", xcheck.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
# -----------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------- TRAINING THE MODEL --------------------------------------------------------------
def compute_loss(y, yhat):
    return tf.keras.losses.mean_squared_error(y, yhat)

### Hyperparameter setting and optimization ###

# Optimization parameters:
num_training_iterations = 100  # Increase this to train longer
batch_size = 5  # Experiment between 1 and 64
seq_length = 50  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
rnn_units = 32  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = get_model(num_units=rnn_units, batch_size=batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_step(x, y):

    with tf.GradientTape() as tape:
        yhat = model(x)
        loss = compute_loss(y, yhat)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss
tf_train_step = tf.function(train_step)

for iter in tqdm(range(num_training_iterations)):
    x_batch, y_batch = get_batch(data, seq_length, batch_size)
    loss = train_step(x_batch, y_batch)

    if iter % 100 == 0:
        model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)

# ----------------------------------- DOING STUFF WITH THE MODEL ---------------------------------------------

model = get_model(rnn_units, batch_size=1)

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_sequence(model, start_sequence, generation_length=1000):

    nstart = start_sequence.shape[0]

    x = tf.expand_dims(start_sequence, 0)

    # empty array to store our results
    seq_generated = np.zeros((start_sequence.shape[0] + generation_length, 2))
    seq_generated[:start_sequence.shape[0]] = start_sequence

    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(nstart, nstart+generation_length)):
        yhat = model(x)
        yhat = tf.squeeze(yhat, 0)

        seq_generated[i] = yhat

        x = tf.expand_dims(yhat, 0)

    return seq_generated

# generate data

newseq = generate_sequence(model, start_sequence=np.array([[-5, 0]]))
plt.plot(newseq[:, 0], newseq[:, 1], 'r')
plt.plot(data[:-1, 0], data[:-1, 1], 'k--')
plt.show()
print(newseq)