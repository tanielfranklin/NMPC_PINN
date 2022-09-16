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
        du = cs.MX.sym('du',nu*self.Hc)
        ysp = cs.MX.sym('du',(ny, 1))
        X = cs.MX.sym('du',(nx, self.Hp+1))
        
        # List of variables
        w=[]
        w+=[du]
        w+=[ysp]
        w+=[cs.reshape(X,(self.Hp+1)*nx,1)]
        print(w[2])
        print(w[2].shape)

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
        init=[]
        init+=[ysp0]
        init+=[du0]
        X0=cs.reshape(cs.repmat(x0, 1, self.Hp+1),(self.Hp+1)*nx,1)
        init+=[X0]
        
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
        
        g=[]
        g+=[X[:, 0]-st_pred] # initial conditions
        lbg=[]
        ubg=[]
        

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
            g+=[eq_cond]
            u += future_du[k*nu:k*nu+2, :]  # update u
            const_y1 = (ymin-y)  # .printme(0)
            const_y2 = (y - ymax)  # .printme(1)

            # g+=[const_y1] #<= 0)
            # g+=[const_y2] #<= 0)
        
        lbg+=[0 for i in range((self.Hp+1)*nx)] #lower bound of equalities       
        ubg+=[0 for i in range((self.Hp+1)*nx)] #uper bound of equalities
        
        # lbg+=[0 for i in range(self.Hp*ny)] #lower bound ymin
        # lbg+=[0 for i in range(self.Hp*ny)] #lower bound ymax
        # ubg+=[np.inf for i in range(self.Hp*ny)] #uper bound ymin
        # ubg+=[np.inf for i in range(self.Hp*ny)] #uper bound ymax
        

        obj = obj_1 + obj_2 + (du.T@R@du)  # complete objective terms

        # # # # Define contraints related to maximum and minimum du rate
        # opti.subject_to(np.tile(self.bcs.dumax, (nu, 1))
        #                 >= du)  # Maximum control rate
        # opti.subject_to(-np.tile(self.bcs.dumax, (nu, 1))
        #                 <= du)  # Minimun control rate

        
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
        opts = {}
        opts['ipopt.tol'] = 1e-1
        opts['ipopt.max_iter'] = 50
        opts["ipopt.constr_viol_tol"]= 1e-1
        
        # Allocate an NLP solver
        
        g_lim=cs.vertcat(*g)
        nlp = {'x':cs.vertcat(*w), 'f':obj, 'g':g_lim}
        # print(nlp)
        solver = cs.nlpsol('solver', 'ipopt', nlp,opts)
        # print(solver)
        x0_init=cs.vertcat(*init)


        
        try:
            sol = solver( # Lower variable bound
             #ubx = ubw,  # Upper variable bound
             lbg = 0,  # Lower constraint bound
             ubg = cs.vertcat(*ubg),  # Upper constraint bound
             x0  = x0_init) # Initial guess
        except RuntimeError:
            print("equalities constraints")
            #print(self.opti.debug.value(st_next-st_pred))
            # print("States variables optimizer")
            # print(self.opti.debug.value(self.bcs.desnorm_x(st_next)))
            # print("States variables model")
            # print(self.opti.debug.value(self.bcs.desnorm_x(st_pred)))
            # print("output variables")
            # print(self.opti.debug.value(y*self.bcs.y_sc))
            print("output variables limits")
            print((ymin*self.bcs.y_sc).T)
            print((ymax*self.bcs.y_sc).T)
            print("decision variables du")
            #print(self.opti.debug.value(du))
            print("decision variables ysp")
            #print(self.opti.debug.value((ysp*self.bcs.y_sc).T))
            print("decision variables now")
            # print(sol.value(X))
            # print(sol.value(ysp))
            # print(sol.value(du))
            # print("decision on initial")
            # print(sol.value(X, opti.initial()))
            # print(sol.value(ysp, opti.initial()))
            # nmpc.opti.debug.show_infeasibilities()
            raise
        w_opt = sol['x'].full().flatten()
        Du = np.vstack(w_opt[:self.bcs.nu*self.Hc])
        ysp_opt = np.vstack(w_opt[self.bcs.nu*self.Hc:self.bcs.nu*self.Hc+ny])

        return Du, ysp_opt*self.bcs.y_sc, sol
