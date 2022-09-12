
from ast import arg
from casadi import MX, SX, fabs, hcat, horzcat, vertcat, Opti, Function, nlpsol, sum1, qpsol, vcat, integrator
from scipy.integrate import solve_ivp
from casadi import sqrt as csqrt
import matplotlib.pyplot as plt
import math
from parameters import Parameters, Lim_c
from param import *
from bcs_envelope import BcsEnvelope


class BCS_model(object):
    def __init__(self, nu, nx, ny, Ts, umin, umax, dumax):
        self.par = Parameters()
        self.Ts, self.umin, self.umax, self.dumax = Ts, umin, umax, dumax
        self.J = None
        self.pm = 20e5
        self.ny = ny
        # self.nu=r.shape[0]
        self.nu = nu
        self.nx = nx
        self.ny = ny
        self.x = MX.sym("x", self.nx)  # Estados
        self.u = MX.sym("u", self.nu)  # Exogena
        self.eq_estado = None
        self.yss = None
        self.y = None
        self.sol = None
        self.estacionario = None
        self.matrizes_ekf = None
        self.sea_nl = None
        self.BCS_equation()
        self.dudt_max = MX.sym("dudt_max", 2)  # Exogena

        self.envelope = BcsEnvelope()

    def norm_x(self, x):
        """Normalize x 
        Args:
            x (_type_): Column array
        Returns:
            _type_: Column array
        """
        if x.shape[1] != 1:
            raise ValueError
        xn = (x-self.par.x0)/self.par.xc
        return xn
    
    def desnorm_x(self, xn):
        """Desnormalize x 
        Args:
            xn (_type_): Column array
        Returns:
            _type_: Column array
        """
        if xn.shape[1] != 1:
            raise ValueError
        x = (xn*self.par.xc+self.par.x0)
        return x

    def BCS_equation(self):
        x = self.x  # Estados
        u = self.u  # Exogena
        # estados
        pbh = x[0]
        pwh = x[1]
        q = x[2]
        fq = x[3]
        zc = x[4]
        # Entradas
        fqref = u[0]
        zcref = u[1]
        pm = self.pm
        pr = 1.26e7
        # pm=u[2]
        # pr=u[3]

        # Calculo do HEAD e delta de press�o
        q0 = ((q*self.par.qc+self.par.qmin) / Cq * (f0 / fq))
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = (CH * H0 * (fq / f0) ** 2)  # Head
        # Pp = rho * g * H  # Delta de press�o

        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3  # Potencia
        I = Inp * P / Pnp  # Corrente

        # Calculo da press�o de intake
        F1 = (0.158 * ((rho * L1 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D1 * A1 ** 2))
              * (mu / (rho * D1 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4))
        F2 = 0.158 * ((rho * L2 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D2 * A2 ** 2)
                      ) * (mu / (rho * D2 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        pin = (pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1)
        #pin = (pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1)
        # Vazao do reservatorio e vazao na choke
        qr = PI * (pr - (pbh*self.par.pbc+self.par.pbmin))
        qch = (zc/100)*Cc * csqrt(fabs(pwh*self.par.pwc+self.par.pwmin - pm))
        F1c = Lim_c(self.par.F1lim)
        F2c = Lim_c(self.par.F2lim)
        Hc = Lim_c(self.par.H_lim)
        qcc = Lim_c(self.par.qch_lim)
        # Normalizar termos não lineares
        ##########################
        qch = (qch-self.par.qch_lim[0])/qcc
        F1 = ((F1-self.par.F1lim[0])/F1c)
        F2 = ((F2-self.par.F2lim[0])/F2c)

        H = (H-self.par.H_lim[0])/Hc
        ###########################

        dpbhdt = (1/self.par.pbc)*b1/V1*(qr - (q*self.par.qc+self.par.qmin))
        dpwhdt = (1/self.par.pwc)*b2/V2 * \
            ((q*self.par.qc+self.par.qmin) - (qcc*qch+self.par.qch_lim[0]))
        dqdt = ((1/(self.par.qc*M))*(pbh*self.par.pbc+self.par.pbmin - (pwh*self.par.pwc+self.par.pwmin) - rho *
                g*hw - (F1c*F1+self.par.F1lim[0]) - (F2c*F2+self.par.F2lim[0]) + rho * g * (H*Hc+self.par.H_lim[0])))
        dfqdt = (fqref - fq)/tp[0]
        dzcdt = (zcref - zc)/tp[1]
        self.dxdt = vertcat(dpbhdt, dpwhdt, dqdt, dfqdt, dzcdt)
        yp = vertcat(pin, (H*Hc+self.par.H_lim[0]))
        self.c_eq_medicao = Function('c_eq_medicao', [x], [yp], ['xn'], ['yp'])
        self.eq_estado = Function('eq_estado', [x, u], [
                                  self.dxdt], ['x', 'u'], ['dxdt'])
        dxdt_0 = self.eq_estado(x, u)
        self.J = sum1(dxdt_0**2)

        # EDO
        # -------------------------------------------------

        opt = {
            'ipopt': {
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
                'max_iter': 50
            },
            'print_time': 0,
        }

        MMQ = {'x': x, 'f': self.J, 'p': u}
        self.solver = nlpsol('solver', 'ipopt', MMQ, opt)
        args = {
            'lbx': np.zeros((self.nx, 1)),
            # m�ximo
            'ubx': np.full((self.nx, 1), np.inf)
        }

    def eq_medicao(self, x):
        pbh = x[0]
        q = x[2]
        fq = x[3]
        # Calculo do HEAD e delta de press�o
        q0 = q / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3  # Potencia
        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * (q) ** 2) / (D1 * A1 ** 2)) * \
            (mu / (rho * D1 * q)) ** (1 / 4)
        pin = pbh - rho * g * h1 - F1
        return vertcat([pin, H])

    def integrator_ode(self, xss, uss):
        xssn = (xss-self.par.x0)/self.par.xc
        #array([0.656882  , 0.58981816, 0.41643043])
        sol = self.solver(x0=xssn, p=uss)
        a = sol['x']
        xn = np.array(sol['x'])
        x = (xn*self.par.xc+self.par.x0)
        return x

    def bcs_model(self, t, x, u):
        # estados
        pbh = x[0]
        pwh = x[1]
        q = x[2]
        fq = x[3]
        zc = x[4]
        # Entradas
        fqref = u[0]
        zcref = u[1]
        pm = self.pm
        pr = 1.26e7
        # pm=u[2]
        # pr=u[3]

        # Calculo do HEAD e delta de press�o
        q0 = (q*self.par.qc+self.par.qmin) / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        # Pp = rho * g * H  # Delta de press�o

        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3  # Potencia
        I = Inp * P / Pnp  # Corrente

        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D1 * A1 ** 2)
                      ) * (mu / (rho * D1 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        F2 = 0.158 * ((rho * L2 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D2 * A2 ** 2)
                      ) * (mu / (rho * D2 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        pin = pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1
        # Vazao do reservatorio e vazao na choke
        qr = PI * (pr - (pbh*self.par.pbc+self.par.pbmin))
        qch = (zc/100)*Cc * math.sqrt(abs(pwh*self.par.pwc+self.par.pwmin - pm))

        # Termos não lineares
        # #menor q implica em menor F
        # funcH=Function('funcH',[self.x,self.u],[H])
        # funcF1=Function('funcF1',[self.x],[F1])
        # funcF2=Function('funcF2',[self.x],[F2])
        # #F1lim=(funcF1([0,0,self.par.qlim[0]]),funcF1([0,0,self.par.qlim[1]]))
        # #F2lim=(funcF2([0,0,self.par.qlim[0]]),funcF2([0,0,self.par.qlim[1]]))
        F1c = Lim_c(self.par.F1lim)
        F2c = Lim_c(self.par.F2lim)
        Hc = Lim_c(self.par.H_lim)
        qcc = Lim_c(self.par.qch_lim)
        # Normalizar termos não lineares
        ##########################
        qch = (qch-self.par.qch_lim[0])/qcc
        F1 = (F1-self.par.F1lim[0])/F1c
        F2 = (F2-self.par.F2lim[0])/F2c
        H = (H-self.par.H_lim[0])/Hc
        ###########################

        dpbhdt = (1/self.par.pbc)*b1/V1*(qr - (q*self.par.qc+self.par.qmin))
        dpwhdt = (1/self.par.pwc)*b2/V2 * \
            ((q*self.par.qc+self.par.qmin) - (qcc*qch+self.par.qch_lim[0]))
        dqdt = (1/(self.par.qc*M))*(pbh*self.par.pbc+self.par.pbmin - (pwh*self.par.pwc+self.par.pwmin) - rho *
                                    g*hw - (F1c*F1+self.par.F1lim[0]) - (F2c*F2+self.par.F2lim[0]) + rho * g * (H*Hc+self.par.H_lim[0]))
        dfqdt = (fqref - fq)/tp[0]
        dzcdt = (zcref - zc)/tp[1]
        dudt_max = [dfq_max, dzc_max]
        dudt = np.zeros((1, 2))
        if abs(dfqdt) > dudt_max[0]:
            dudt[0] = np.sign(dfqdt)*dudt_max[0]
        else:
            dudt[0] = dfqdt
        if (abs(dzcdt) > dudt_max[1]):
            dudt[1] = np.sign(dzcdt)*dudt_max[1]
        else:
            dudt[1] = dzcdt
        dxdt = np.vstack(dpbhdt, dpwhdt, dqdt, dudt[0], dudt[1])
        return dxdt

    def open_loop_sim(self, x0m, du, uk_1, pm):
        y = []
        for k in range(self.Hp):
            uk_1 = uk_1 + du[(k)*self.nu:2+k*self.nu, :]
            xk = self.integrator_ode(x0m, uk_1)
            x0m = xk
            ymk = self.eq_medicao(x0m)  # dimension ny
            y.append(ymk)  # dimension Hp*ny
        y = vcat(y)
        return y
    # def c_open_loop_sim(self,x0m,du,uk_1,pm):
    #     y=[]
    #     for k in range(self.Hp):
    #         uk_1 = uk_1 + du[(k)*self.nu:2+k*self.nu,:]
    #         xk=self.integrator_ode(x0m,uk_1)
    #         x0m = xk
    #         ymk = self.c_eq_medicao(x0m); # dimension ny
    #         y.append(ymk) # dimension Hp*ny
    #     y=vertcat(y)
        # return y

    def RK_integrator(self, x, u):
        sol = solve_ivp(self.bcs_model, [0, self.Ts], [x, u], method='RK45')
        sol.y[0]

    def eq_medicao(self, x):
        pbh = x[0]
        q = x[2]
        fq = x[3]
        # Calculo do HEAD e delta de press�o
        q0 = q / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3  # Potencia
        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * (q) ** 2) / (D1 * A1 ** 2)) * \
            (mu / (rho * D1 * q)) ** (1 / 4)
        pin = pbh - rho * g * h1 - F1
        return vertcat(pin, H)
