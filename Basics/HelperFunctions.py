import numpy as np
import tensorflow as tf
import LCSolver

# get a random batch of training data from a dataset
def get_batch(dataset, sequence_length, batch_size):
    # length of the dataset (wo last element, so we dont go over bound when calculating y)
    n = dataset.shape[0] - 1
    
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - sequence_length, batch_size) # list of indices of lenght batch_size

    # get batches (list of np.arrays of length batch_size, and each np.array is of the shape (sequence_lenght, 2))
    x_batch = np.array([dataset[ind:ind+sequence_length] for ind in idx])
    y_batch = np.array([dataset[ind+1:ind+sequence_length+1] for ind in idx]) # same as input_batch, but moved 1 to the right

    #np.testing.assert_allclose(x_batch, np.array(input_batch))

    return x_batch, y_batch




# test functions
if __name__ == '__main__':

    lc = LCSolver.Simple2DLimitCycle(x1c=0, x2c=0)
    data = lc.solve_nstep(
        x0=np.array([-1,0]),
        t0=0,
        tStep=0.1,
        nStep=49
    )

    print("\n\n Testing 'get_batch(data, 5, 2)':")
    print(get_batch(data, 5, 2))