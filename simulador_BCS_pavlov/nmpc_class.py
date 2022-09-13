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
        self.NMPC_eq()
    def RK_ode_integrator(self,x0,u0):
        DT=self.bcs.Ts
        k1 = self.bcs.eq_estado(x0, u0)
        k2 = self.bcs.eq_estado(x0 + DT/2 * k1, u0)
        k3 = self.bcs.eq_estado(x0 + DT/2 * k2, u0)
        k4 = self.bcs.eq_estado(x0 + DT * k3, u0)
        X_next = x0+DT/6*(k1 + 2*k2 + 2*k3 + k4)
        X0 = X_next
        return x0

    def NMPC_eq(self):
        nx = self.bcs.nx
        ny = self.bcs.ny
        nu = self.bcs.nu
        self.opti = cs.Opti()
        # Decision Variables must be keep free
        du = self.opti.variable(nu*self.Hc, 1)
        #################################
        # self.du= np.zeros((self.Hc*self.bcs.nu, 1)) # keep commented
        # predictions of x along trajectory
        X0= self.opti.variable(nx,1)
        u0= self.opti.variable(nu,1)
        X = self.opti.variable(nx, self.Hp+1)
        # predictions of y along trajectory
        Y = self.opti.variable(ny, self.Hp+1)
                
        # normalizing the first value for predictions
        X0 = self.bcs.norm_x(X0)
        Y0 = self.bcs.c_eq_medicao(X0)  # predictions of y
               
        # Future control increments along Hp
        #u0=np.ones((2,1))*50
        #X0 = np.vstack([0.656882, 0.58981816, 0.41643043, 50.0, 50.0])
        future_du = cs.vertcat(du, np.zeros((nu*(self.Hp-self.Hc), 1)))
        # starting filling prediction matrix
        X_pred = [X0]
        Y_pred = [Y0]
        Uk =[u0]
    

        # Filling predictions matrix with values
        print("Creating predictions matrices")
        for k in range(self.Hp):
            u0 += future_du[k*nu:k*nu+2, :] #applying exogenous inputs
            ## using integrator
            sol = self.bcs.F(x0=X0, p=u0)
            X_next = cs.vertcat(sol['xf'])
            ##
            Uk.append(u0)
            X0 = X_next
            X_pred.append(X_next)
            y_next = self.bcs.c_eq_medicao(X_next)
            Y_pred.append(y_next )

        X = cs.hcat(X_pred)
        Y = cs.hcat(Y_pred)
        U = cs.hcat(Uk)

        self.FF = cs.Function('FF', [du, X0,u0], [X, Y, U], ['du',
                              'x0','u0'], ['X', 'Y', 'U'])

    def nmpc_solver(self, P, ymin, ymax):
        # P=np.vstack([x0,Du,utg,uk_1,yss])
        nx = self.bcs.nx
        ny = self.bcs.ny
        nu = self.bcs.nu
        Qu = np.diag(np.array([self.qu]))  # Weights control effort 
        Q = np.diag(np.tile(self.q[:,0],self.Hp)) # Weights setpoint error
        Q = np.diag(self.q[:,0]) # Weights setpoint error
        R = np.diag(np.tile(self.r, (1, self.Hc))[0, :]) # Weights economic target error
        opti = self.opti
        # define decision variables
        du = opti.variable(nu*self.Hc)
        ysp = opti.variable(ny,1)
        x0 = P[:nx]  # Get X0 from P
        x0 = self.bcs.norm_x(x0) # Normalize states x0

        len_du = nu*self.Hc
        utg = P[nx+len_du:nx+len_du+1, :]  # Get economic target from P
        # Get initial control actions from P
        ysp0 = P[-ny:, :] 
        du0 = P[nx+len_du+1+nu:nx+len_du+1+nu+len_du]
        # Initialize decision variables
        opti.set_initial(ysp, ysp0)
        opti.set_initial(du, du0)
        # Recovering predictions of states and outputs matrices
        X, Y, u = self.FF(du, P)

        # #Define dynamic constraints  dependent of predictions steps

        obj_1,obj_2=0,0
        for k in range(self.Hp):
            opti.subject_to(X[:, k+1]-X[:, k]==0)
            # opti.subject_to(Y[:, k+1] >= ymin)
            # opti.subject_to(Y[:, k+1] <= ymax)
            # Track ysp_opt
            obj_1+=(Y[:,k+1]-ysp).T@Q@(Y[:,k+1]-ysp)
            # Track Maximum zc
            obj_2+=((u[1] - utg).T*Qu*(u[1] - utg))

        obj = obj_1 + obj_2+(du.T@R@du)
        
        # # Define contraints related to maximum and minimum du rate
        # opti.subject_to(np.tile(self.bcs.dumax, (nu, 1))
        #                 >= du)  # Maximum control rate
        # opti.subject_to(-np.tile(self.bcs.dumax, (nu, 1))
        #                 <= du)  # Minimun control rate

 
        opti.minimize(obj)
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        # opts = {'ipopt.print_level': 3}
        opti.solver("ipopt")
        sol = opti.solve()
        Du = np.vstack(sol.value(du))
        ysp_opt = np.vstack(sol.value(ysp)[:ny])
        return Du,ysp_opt
        # # opti.subject_to(du+M_dumax>=0)
        # # opti.subject_to(-du+M_dumax>=0)
        # p_opts = {"expand": True}
        # s_opts = {"max_iter": 80, "print_level": 3}
        # opti.solver("ipopt", p_opts, s_opts)
        # print("----------------------------------------------------------------")
        # # print("Restrições com problema:")
        # # print(opti.debug.g_describe(73))
        # # print("Valores com problema:")
        # # #print(opti.debug.value(ynext))
        # # print("Variáveis com problema:")
        # # print(opti.debug.x_describe(0))
        # print("----------------------------------------------------------------")
        # sol = opti.solve()
        # # opti.debug.show_infeasibilities()
        # #print("Valores com problema:",opti.debug.g_describe(0),opti.debug.x_describe(0))
        # # #print(S.stats())
        # # x_opt = r['x']
        # Du = sol.value(du)
        # Du = Du.reshape((Du.shape[0], 1))
        # ysp_opt = sol.value(ysp)[:ny]
        # ysp_opt = ysp_opt.reshape((ysp_opt.shape[0], 1))
        # return Du, ysp_opt
