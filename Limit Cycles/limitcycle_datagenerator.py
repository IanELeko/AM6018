import numpy as np
import matplotlib.pyplot as plt
import limitcycle
import tensorflow.keras as keras

class LimitCycleDataGenerator():

    def __init__(self, x0, t_step, rk_steps, seq_len):
        self.x0 = x0
        self.t_step = t_step
        self.rk_steps = rk_steps
        self.seq_len = seq_len

    def generate_data(self):
        lc = limitcycle.SimpleLimitCycle()
        data = lc.solve_nstep(
            x0=self.x0, 
            t0=0, 
            tStep=self.t_step, 
            nStep=self.rk_steps
        )

        return data

    def generate_time_series(self):
        data = self.generate_data()
        ts_data = keras.utils.timeseries_dataset_from_array(
            data=data[:-1], 
            targets=data[self.seq_len:], 
            sequence_length=self.seq_len
        )

        return ts_data
