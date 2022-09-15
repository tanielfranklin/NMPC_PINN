from BCS_casadi import BCS_model
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt


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
        yss=self.bcs.yss
        uss=self.bcs.uss
        

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
        u0 = P[nx:nx+nu]/uss
        utg = P[-1]/uss[1]  # Get economic target from P
        ysp0 = self.bcs.c_eq_medicao(x0)/yss
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
        print("controle: ",u)
        eps = 0.05
        opti.subject_to(X[:, 0] == self.bcs.F(x0=x0, p=u0*uss)['xf'])
        #print(self.bcs.F(x0=x0, p=u0)['xf'].T)
        for k in range(self.Hp):
            st = X[:, k]  # states
            y = self.bcs.c_eq_medicao(st)/yss  # system outputs
            obj_1 += (y-ysp).T@Q@(y-ysp)  # set points terms
            # Track economic target (Maximum zc)
            obj_2 += ((u[1] - utg).T*Qu*(u[1] - utg))
            st_next = X[:, k+1]
            st_pred=self.bcs.F(x0=st, p=u*uss)['xf']
            #eq_cond=(st_next-st_pred).printme(0)
            eq_cond=(st_next-st_pred)
            opti.subject_to(eq_cond==0)
            u += future_du[k*nu:k*nu+2, :]  # update u

            opti.subject_to(y >= ymin/yss)
            opti.subject_to(y <= ymax/yss)
        # print(ymin.T+eps)
        # print(ymax.T+eps)

        obj = obj_1 + obj_2+(du.T@R@du)  # complete objective terms

        # # # Define contraints related to maximum and minimum du rate
        opti.subject_to(np.tile(self.bcs.dumax, (nu, 1))
                        >= du)  # Maximum control rate
        opti.subject_to(-np.tile(self.bcs.dumax, (nu, 1))
                        <= du)  # Minimun control rate

        opti.minimize(obj)
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opts = {'ipopt.print_level': 3}
        p_opts = {"expand": True, "print_time": False}
        s_opts = {#"max_cpu_time": 0.1,
                  "print_level": 0,
                  "tol": 5e-1,
                #   "dual_inf_tol": 5.0,
                  "constr_viol_tol": 1e-1,
                #   "compl_inf_tol": 1e-1,
                #   "acceptable_tol": 1e-2,
                #   "acceptable_constr_viol_tol": 0.01,
                #   "acceptable_dual_inf_tol": 1e10,
                  "acceptable_compl_inf_tol": 0.01,
                  "acceptable_obj_change_tol": 1e20,
                  "diverging_iterates_tol": 1e20}
        # print(opti.debug.show_infeasibilities())
        s_opts = {"print_level": 5,"max_iter":80,
                  #"nlp_scaling_method":None
                  "tol": 5e-1,
                  "constr_viol_tol": 1e-1,
         }
        opts = {}
        opti.solver("ipopt",opts, s_opts)
        #opti.solver('sqpmethod',{'qpsol':'osqp'})
        self.opti=opti
        # plt.figure(1)
        # opti.callback(lambda i: plt.plot(opti.debug.value(x0[0])))
        # plt.show()
        sol = opti.solve()
        # print(self.opti.debug.value(st_next-st_pred))
        # print(self.opti.debug.value(st_next))
        
        
        Du = np.vstack(sol.value(du))
        ysp_opt = np.vstack(sol.value(ysp)[:ny])

        return Du, ysp_opt*yss, sol
