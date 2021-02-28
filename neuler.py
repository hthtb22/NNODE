import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F


class NEuler:
    def __init__(self, f, x0, N, T):
        self.f, self.x0, self.N, self.T = f, x0, N, T
        self._Dx = torch.tensor(sp.diags([[1] * N, [-1] * N], [0, -1], shape=(N, N)).A, device="cuda").float()

        ts_pre = np.linspace(0, T, N + 1)
        ss_pre = 0 * ts_pre

        xj = x0
        for j, tj in enumerate(ts_pre):
            ss_pre[j] = f(tj, xj)
            xj = xj + T / N * ss_pre[j]

        # Tanulando parameterek
        self._ts = torch.tensor(ts_pre[1:-1].reshape((N - 1, 1)), dtype=torch.float, requires_grad=True, device="cuda")
        self.ss = torch.tensor(ss_pre[1:].reshape((N, 1)), requires_grad=True, dtype=torch.float, device="cuda")

    def approximate(self, ts, ss):
        """ approximate :: (ts, ss) -> xs ~ x(ts) """
        return F.relu(ts[1:] - ts[:-1].reshape(1, self.N)) @ self._Dx @ ss + self.x0

    def gauss_quadrature2(self, t1, t2, x1, x2):
        rs = [(1 - 1/3**.5)/2, (1 + 1/3**.5)/2]
        ws = [.5, .5]
        s = 0
        for r, w in zip(rs, ws):
            s += w * self.f(r * t1 + (1-r)*t2, r*x1 + (1-r)*x2)

        return s

    def loss_function(self, ts, xs):
        """
            loss :: (ts, xs) -> err
        """
        err = torch.sum(torch.square(xs[0] - self.x0 - (ts[1] - 0) * self.gauss_quadrature2(0, ts[1], self.x0, xs[0])))

        for t1, t2, x1, x2 in zip(ts[1:], ts[2:], xs, xs[1:]):
            err += torch.sum(torch.square(x2 - x1 - (t2-t1)*self.gauss_quadrature2(t1, t2, x1, x2)))

        return err

    def solve(self, max_it=2000, lr=.005, atol=1e-2):
        optimizer = torch.optim.SGD((self._ts, self.ss), lr=lr)

        for _ in range(max(max_it, 1)):
            optimizer.zero_grad()
            # idopontok pad-olasa
            ts = F.pad(F.pad(self._ts, (0, 0, 1, 0), "constant", 0),
                                       (0, 0, 0, 1), "constant", self.T)
            xs = self.approximate(ts, self.ss)
            loss = self.loss_function(ts, xs)
            if _ % 100 == 0:
                print(f"step is {_}. Loss is {loss.item()}")
                if loss.item() <= atol:
                    break

            loss.backward()
            optimizer.step()

        return ts.cpu().detach().numpy()[1:].T[0], xs.cpu().detach().numpy().T[0]


def main():
    # ode jobboldal
    def f(t, x):
        return -22 * (t - 1) * x

    # kezdeti feltetel
    x0 = np.exp(-7)

    # intervallum [0, T]
    T = 2

    # szakaszok szama
    N = 40

    num_method = NEuler(f, x0, N, T)
    ts, xs = num_method.solve(max_it=2000, lr=0.0005)

    plt.plot(ts, xs, 'o-')
    print(f"ts: {ts}\nxs:{xs}")
    plt.show()


if __name__ == '__main__':
    main()