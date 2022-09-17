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

    def nmpc_solver(self, P, ysp0, ylim):

        uss = self.bcs.uss
        ymin = ylim[0]/self.bcs.y_sc   # normalize
        ymax = ylim[1]/self.bcs.y_sc   # normalize
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
        ysp0 = ysp0/self.bcs.y_sc
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
        try:
            st_pred = self.bcs.F(x0=x0, p=u0*uss)['xf']
        except Exception as e:
            print("An exception was raised in integrator computing:")
            print(e)
            raise
        print("controle: ", (u*uss).T)
        print("Estados: ", self.bcs.desnorm_x(x0).T)
        print("Saídas: ", (ysp0*self.bcs.y_sc).T)
        print("Estados preditos: ", self.bcs.desnorm_x(st_pred).T)
        print("Limites Saída: ", (ymin*self.bcs.y_sc).T)
        print("Limites Saída: ", (ymax*self.bcs.y_sc).T)

        opti.subject_to(X[:, 0] == st_pred)
        #print(self.bcs.F(x0=x0, p=u0)['xf'].T)
        for k in range(self.Hp):
            st = X[:, k]  # states
            y = self.bcs.c_eq_medicao(st)/self.bcs.y_sc  # system outputs
            obj_1 += (y-ysp).T@Q@(y-ysp)  # set points terms
            # Track economic target (Maximum zc)
            obj_2 += ((u[1] - utg).T*Qu*(u[1] - utg))
            st_next = X[:, k+1]
            try:
                st_pred = self.bcs.F(x0=st, p=u*uss)['xf']
            except Exception as e:
                print("An exception was raised in integrator II índice: ", k)
                print(e)
                raise
            eq_cond = (st_next-st_pred)  # .printme(0)
            opti.subject_to(eq_cond == 0)
            u += future_du[k*nu:k*nu+2, :]  # update u
            const_y1 = (y - ymin)  # .printme(0)
            const_y2 = (y - ymax)  # .printme(1)

            # opti.subject_to(const_y1 >= 0)
            # opti.subject_to(const_y2 <= 0)
        # print(ymin.T+eps)
        # print(ymax.T+eps)

        obj = obj_1 + obj_2  +(du.T@R@du)  # complete objective terms

        # # # # Define contraints related to maximum and minimum du rate
        opti.subject_to(np.tile(self.bcs.dumax, (nu, 1))
                        >= du)  # Maximum control rate
        opti.subject_to(-np.tile(self.bcs.dumax, (nu, 1))
                        <= du)  # Minimun control rate

        opti.minimize(obj)
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opts = {'ipopt.print_level': 3}
        p_opts = {"expand": True, "print_time": False}
        s_opts = {  # "max_cpu_time": 0.1,
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
        s_opts = {"print_level": 5, "max_iter": 50,
                  # "nlp_scaling_method":None,
                  "tol": 1e-1,
                  "constr_viol_tol": 5e-1,
                  }
        opts = {}
        opti.solver("ipopt", opts, s_opts)
        # opti.solver('sqpmethod',{'qpsol':'osqp'})
        self.opti = opti
        # plt.figure(1)
        # opti.callback(lambda i: plt.plot(opti.debug.value(x0[0])))
        # plt.show()
        try:
            sol = opti.solve()
        except RuntimeError:
            print("equalities constraints")
            print(self.opti.debug.value(st_next-st_pred))
            print("States variables optimizer")
            print(self.opti.debug.value(self.bcs.desnorm_x(st_next)))
            print("States variables model")
            print(self.opti.debug.value(self.bcs.desnorm_x(st_pred)))
            print("output variables")
            print(self.opti.debug.value(y*self.bcs.y_sc))
            print("output variables limits")
            print((ymin*self.bcs.y_sc).T)
            print((ymax*self.bcs.y_sc).T)
            print("decision variables du")
            print(self.opti.debug.value(du))
            print("decision variables ysp")
            print(self.opti.debug.value((ysp*self.bcs.y_sc).T))
            print("decision variables now")
            print(sol.value(X))
            print(sol.value(ysp))
            print(sol.value(du))
            print("decision on initial")
            print(sol.value(X, opti.initial()))
            print(sol.value(ysp, opti.initial()))
            # nmpc.opti.debug.show_infeasibilities()
            raise

        Du = np.vstack(sol.value(du))
        ysp_opt = np.vstack(sol.value(ysp)[:ny])

        return Du, ysp_opt*self.bcs.y_sc, sol
    
    
