import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

class LinearSys():

    def __init__(self, A):
        self.A = A
        self.odefun = lambda t, x: np.dot(self.A, x)
        self.solver = None

    def get_solver(self, x0, t0, tN, tStep):
        self.solver = scipy.integrate.RK45(
            fun=self.odefun,
            t0=t0,
            y0=x0,
            t_bound=tN,
            first_step=tStep,
            max_step=tStep
        )
    
    def solve(self, x0, t0, tN, tStep):
        self.get_solver(x0, t0, tN, tStep)

        sol = np.zeros((int((tN - t0) / tStep) + 1, x0.shape[0]))

        i = 0
        while self.solver.status != 'finished':
            sol[i] = self.solver.y
            i += 1
            self.solver.step()
        
        print(i)
        print(sol.shape)
        return sol
        
if __name__ == '__main__':
    ls = LinearSys(np.array([[-2.28, 1.95], [-2.25, 1.48]]))

    x1 = np.linspace(-1, 1, 3)
    x2 = np.linspace(-1, 1, 3)
    x1v, x2v = np.meshgrid(x1, x2)

    for i in range(len(x1)):
        for j in range(len(x2)):
            x0 = np.array([x1v[i, j], x2v[i, j]])
            xs = ls.solve(x0, 0, 100, 0.1)
            print(f'i = {i}, j = {j}\n xs={xs}')
            plt.plot(xs[:, 0], xs[:, 1])

    plt.show()

'''def LotkaVolterraODE(x, coef):
    a, b, c, d = coef[0], coef[1], coef[2], coef[3]
    x1, x2 = x[0], x[1]

    dx1dt = a*x1 - b*x1*x2
    dx2dt = d*x1*x2 - c*x2

    return np.array([dx1dt, dx2dt])

def LotkaVolterra1(t, x):
    return LotkaVolterraODE(x, (1, 1, 1, 1))

x0 = np.array([2, 2])

t0, tN, tStep = 0, 10, 0.1
xs = np.zeros((int((tN - t0) / tStep) + 1, 2))
xs[0] = x0

sol = scipy.integrate.RK45(
    fun=LotkaVolterra1,
    t0=t0,
    y0=x0,
    t_bound=tN,
    first_step=tStep,
    max_step=tStep
)

i = 0
while sol.status != 'finished':
    xs[i] = sol.y
    i += 1
    sol.step()

plt.plot(xs[:, 0], xs[:, 1])
plt.show()

print(xs)'''

