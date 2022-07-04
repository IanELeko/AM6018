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

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = createLSTM.get_model(createLSTM.rnn_units, batch_size=1)

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()