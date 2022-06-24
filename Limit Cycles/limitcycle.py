import scipy.integrate
import numpy as np
import tensorflow as tf

class SimpleLimitCycle():

    def __init__(self, x1c=0, x2c=0):
        self.x1c, self.x2c = x1c, x2c
        self.centre = (x1c, x2c)
    
    def odefun(self, t, x):
        assert(len(x) == 2)
        x1, x2 = x[0] - self.x1c, x[1] - self.x2c

        r2 = x1**2 + x2**2

        x1dot = -x2 + x1*(1 - r2)
        x2dot = +x1 + x2*(1 - r2)

        return np.array([x1dot, x2dot])


    def get_solver(self, x0, t0, tN, tStep):
        self.solver = scipy.integrate.RK45(
            fun=self.odefun,
            t0=t0,
            y0=x0,
            t_bound=tN,
            first_step=tStep,
            max_step=tStep
        )

    def solve_tstep(self, x0, t0, tN, tStep):
        self.get_solver(x0, t0, tN, tStep)

        sol = np.zeros((int((tN - t0) / tStep) + 1, x0.shape[0]))

        i = 0
        while self.solver.status != 'finished' and i < sol.shape[0]:
            sol[i] = self.solver.y
            i += 1
            self.solver.step()
        
        return sol

    def solve_nstep(self, x0, t0, tStep, nStep):
        sol = self.solve_tstep( 
            x0=x0, 
            t0=t0, 
            tN=t0 + tStep*nStep, 
            tStep=tStep
        )

        return sol

class SimpleLimitCycleDataGenerator():

    def __init__(self, x0, t_step, rk_steps, seq_len):
        self.x0 = x0
        self.t_step = t_step
        self.rk_steps = rk_steps
        self.seq_len = seq_len

    def generate_data(self):
        lc = SimpleLimitCycle()
        data = lc.solve_nstep(
            x0=self.x0, 
            t0=0, 
            tStep=self.t_step, 
            nStep=self.rk_steps
        )

        return data

    def generate_time_series(self):
        data = self.generate_data()
        ts_data = tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1], 
            targets=data[self.seq_len:], 
            sequence_length=self.seq_len
        )

        return ts_data

class LCDataset():
    def __init__(self):
        pass

    def generate_and_save_dataset(self, x0, T_STEP, RK_STEPS, SEQ_LEN):
        lcgen = SimpleLimitCycleDataGenerator(
            x0=x0,
            t_step=T_STEP,
            rk_steps=RK_STEPS,
            seq_len=SEQ_LEN
        )

        data = lcgen.generate_data()

        np.savetxt(
            fname=f'C:\\Users\\lekoi\\Documents\\AM6018\\Limit Cycles\\Data\\x0={x0},t={T_STEP},rk={RK_STEPS},seq={SEQ_LEN}.csv',
            X=data,
            delimiter=','
        )
    
    def generate_and_save_time_series(self, x0, T_STEP, RK_STEPS, SEQ_LEN):
        lcgen = SimpleLimitCycleDataGenerator(
            x0=x0,
            t_step=T_STEP,
            rk_steps=RK_STEPS,
            seq_len=SEQ_LEN
        )

        data = lcgen.generate_time_series()

        tf.data.experimental.save(
            dataset=data, 
            path=f'C:\\Users\\lekoi\\Documents\\AM6018\\Limit Cycles\\Data\\TSDataset\\', 
            compression=None, 
            shard_func=None, 
            checkpoint_args=None
        )


    def load_dataset(self, fname):
        return np.loadtxt(fname, delimiter=',')
    
    def load_time_series(self, fname):
        return tf.data.experimental.load(
            path=fname, 
            element_spec=None, 
            compression=None, 
            reader_func=None
        )



        