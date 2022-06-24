import numpy as np
import limitcycle

lcd = limitcycle.LCDataset()

x0 = np.array([2, 2])
T_STEP = 0.01
RK_STEPS = 10000
SEQ_LEN = 250

lcd.generate_and_save_dataset(x0, T_STEP, RK_STEPS, SEQ_LEN)
lcd.generate_and_save_time_series(x0, T_STEP, RK_STEPS, SEQ_LEN)