from BCS_casadi import BCS_model
import casadi as cs
import numpy as np


class NMPC(object):

    def __init__(self, Hp, Hc, q, r, qu, bcs_init):
        self.Hp, self.Hc, self.q, self.r, self.qu = Hp, Hc, q, r, qu
        nu, nx, ny, Ts, umin, umax, dumax = bcs_init
        self.bcs = BCS_model(nu, nx, ny, Ts, umin, umax, dumax)
        self.Dumin = lambda ymin: np.vstack(
            [np.tile(-self.dumax, (self.Hc, 1)), ymin])
        self.Dumax = lambda ymax: np.vstack(
            [np.tile(self.dumax, (self.Hc, 1)), ymax])
        # self.NMPC_eq()

    def nmpc_solver(self, P, ymin, ymax):

        opti = cs.Opti()
        nx = self.bcs.nx
        ny = self.bcs.ny
        nu = self.bcs.nu
        Qu = np.diag(np.array([self.qu]))  # Weights control effort
        Q = np.diag(self.q[:, 0])  # Weights setpoint error
        # Weights economic target error
        R = np.diag(np.tile(self.r, (1, self.Hc))[0, :])

        # define decision variables
        du = opti.variable(nu*self.Hc)
        ysp = opti.variable(ny, 1)
        X = opti.variable(nx, self.Hp+1)

        # Parameters vector
        # [x0 u0 du utg]
        x0 = P[:nx]  # Get X0 from P
        x0 = self.bcs.norm_x(x0)  # Normalize states x0
        u0 = P[nx:nx+nu]
        utg = P[-1]  # Get economic target from P
        ysp0 = self.bcs.c_eq_medicao(x0)
        # Get initial and updated control actions from P
        du0 = P[-nu*self.Hc-1:-1]

        # Initialize decision variables
        opti.set_initial(ysp, ysp0)
        opti.set_initial(du, du0)
        opti.set_initial(X, cs.repmat(x0, 1, self.Hp+1))  # fill X with x0

        # Define dynamic constraints  dependent of predictions steps
        future_du = cs.vertcat(du, np.zeros((nu*(self.Hp-self.Hc), 1)))

        # Initialize objective functions
        obj_1, obj_2 = 0, 0
        u = u0
        opti.subject_to(X[:, 0] == self.bcs.F(x0=x0, p=u0)['xf'])
        for k in range(self.Hp):
            st = X[:, k]  # states
            y = self.bcs.c_eq_medicao(st)  # system outputs
            obj_1 += (y-ysp).T@Q@(y-ysp)  # set points terms
            # Track economic target (Maximum zc)
            obj_2 += ((u[1] - utg).T*Qu*(u[1] - utg))
            st_next = X[:, k+1]

            opti.subject_to(st_next == self.bcs.F(x0=st, p=u)['xf'])
            u += future_du[k*nu:k*nu+2, :]  # update u

            # opti.subject_to(y >= ymin)
            # opti.subject_to(y <= ymax)

        obj = obj_1 + obj_2+(du.T@R@du)  # complete objective terms

        # # # Define contraints related to maximum and minimum du rate
        opti.subject_to(np.tile(self.bcs.dumax, (nu, 1))
                        >= du)  # Maximum control rate
        opti.subject_to(-np.tile(self.bcs.dumax, (nu, 1))
                        <= du)  # Minimun control rate

        opti.minimize(obj)
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opts = {'ipopt.print_level': 3}
        opts = {}
        opti.solver("ipopt", opts)
        sol = opti.solve()
        Du = np.vstack(sol.value(du))
        ysp_opt = np.vstack(sol.value(ysp)[:ny])
        return Du, ysp_opt
