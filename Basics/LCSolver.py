import scipy.integrate
import numpy as np

class Simple2DLimitCycle():

    def __init__(self, x1c=0, x2c=0):
        self.x1c, self.x2c = x1c, x2c
    
    def odefun(self, t, x):
        assert(len(x) == 2) # only support 2D eqns
        x1, x2 = x[0] - self.x1c, x[1] - self.x2c # initialise to the centre

        r2 = x1**2 + x2**2

        # ODE eqn
        x1dot = -x2 + x1*(1 - r2)
        x2dot = +x1 + x2*(1 - r2)

        return np.array([x1dot, x2dot])

    def solve_nstep(self, x0, t0, tStep, nStep):
        # initialise solver
        self.solver = scipy.integrate.RK45(
            fun=self.odefun,
            t0=t0,
            y0=x0,
            t_bound=t0 + tStep * nStep + 10, # +10 to make sure that we can iterate nStep steps
            first_step=tStep,
            max_step=tStep
        )

        # initialise the solution (1 initial condition + nStep solutions to the ODE)
        sol = np.zeros((nStep, 2))

        # iterate over the solver
        for i in range(nStep):
            sol[i] = self.solver.y
            i += 1
            self.solver.step()

        return sol

