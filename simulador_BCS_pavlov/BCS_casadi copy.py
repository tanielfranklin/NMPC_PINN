
from casadi import MX,SX,fabs,vertcat, Opti, Function, nlpsol,sum1,qpsol

from casadi import sqrt as csqrt
import math
from parameters import Parameters, Lim_c
from param import *
from bcs_envelope import BcsEnvelope
from plot_result import PlotResult





class BCS_model(object):
    def __init__(self,nx,ny,Hp,Hc,Ts,umin,umax,dumax,q,r,qu):
        self.par=Parameters()
        self.Hp,self.Hc,self.Ts,self.umin,self.umax,self.dumax=Hp,Hc,Ts,umin,umax,dumax
        self.J=None
        self.pm=20e5
        self.qu=qu
        self.ny=q.shape[0]
        # self.nu=r.shape[0]
        self.nu=2
        self.nx = nx
        self.q = q
        self.ny = 2
        self.r = r # weights on control actions
        
        self.x = MX.sym("x",self.nx); # Estados
        self.u = MX.sym("u",self.nu); # Exogena
        self.eq_estado=None
        self.yss=None
        self.y=None
        self.sol=None
        self.estacionario=None
        self.matrizes_ekf=None
        self.sea_nl=None
        self.BCS_equation()
        self.dudt_max = MX.sym("dudt_max",2); # Exogena
        self.Ain,self.Bin = None,None
        
        self.constraints()
        self.envelope=BcsEnvelope()
        
    def constraints(self):
        # Auxiliary matrices for using in the solver
        self.Dumin = lambda ymin: np.vstack([np.tile(-self.dumax,(self.Hc,1)), ymin])
        self.Dumax = lambda ymax: np.vstack([np.tile(self.dumax,(self.Hc,1)), ymax])
      

        Mtil=[]
        Itil=[]
        auxM=np.zeros((self.nu,self.Hc*self.nu))
        for i in range(self.Hc):
            auxM=np.hstack([np.eye(self.nu), auxM[:,0:(self.Hc-1)*self.nu]])
            Mtil.append(auxM)
            Itil.append(np.eye(self.nu))
            
        Mtil=np.vstack(Mtil)
        Itil=np.vstack(Itil)

        # Ain = [Mtil;-Mtil];
        print(Mtil.shape)
        
        self.Ain = np.vstack([np.hstack([Mtil, np.zeros((self.Hc*self.nu,self.ny))]),np.hstack([-Mtil,np.zeros((self.Hc*self.nu,self.ny))])])
        self.Bin = lambda uk_1: np.vstack([np.tile(self.umax,(self.Hc,1)) - Itil@uk_1, Itil@uk_1 - np.tile(self.umin,(self.Hc,1))])
        # Aeq = []
        # beq = []

        

    def BCS_equation(self):
        x = self.x  # Estados
        u = self.u  # Exogena
        #estados
        pbh = x[0]
        pwh = x[1]
        q = x[2]
        fq=x[3]
        zc=x[4]
        # Entradas
        fqref = u[0]
        zcref = u[1]
        pm=self.pm
        pr=1.26e7
        # pm=u[2]
        # pr=u[3]
        
        # Calculo do HEAD e delta de press�o
        q0 = ((q*self.par.qc+self.par.qmin) / Cq * (f0 / fq))
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = (CH * H0 * (fq  / f0) ** 2) # Head
        #Pp = rho * g * H  # Delta de press�o

        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3;  # Potencia
        I = Inp * P / Pnp  # Corrente

        # Calculo da press�o de intake
        F1 = (0.158 * ((rho * L1 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4))
        F2 = 0.158 * ((rho * L2 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D2 * A2 ** 2)) * (mu / (rho * D2 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        pin = (pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1)
        #pin = (pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1)
        # Vazao do reservatorio e vazao na choke
        qr = PI * (pr - (pbh*self.par.pbc+self.par.pbmin))
        qch = (zc/100)*Cc * csqrt(fabs(pwh*self.par.pwc+self.par.pwmin - pm))
        F1c=Lim_c(self.par.F1lim)
        F2c=Lim_c(self.par.F2lim)
        Hc=Lim_c(self.par.H_lim)
        qcc=Lim_c(self.par.qch_lim)
        #Normalizar termos não lineares
        ##########################
        qch=(qch-self.par.qch_lim[0])/qcc
        F1=((F1-self.par.F1lim[0])/F1c)
        F2=((F2-self.par.F2lim[0])/F2c)
        
        H=(H-self.par.H_lim[0])/Hc
        ###########################

        dpbhdt = (1/self.par.pbc)*b1/V1*(qr - (q*self.par.qc+self.par.qmin))
        dpwhdt = (1/self.par.pwc)*b2/V2*((q*self.par.qc+self.par.qmin) - (qcc*qch+self.par.qch_lim[0]))
        dqdt = ((1/(self.par.qc*M))*(pbh*self.par.pbc+self.par.pbmin - (pwh*self.par.pwc+self.par.pwmin) - rho*g*hw - (F1c*F1+self.par.F1lim[0]) - (F2c*F2+self.par.F2lim[0]) + rho * g * (H*Hc+self.par.H_lim[0])))
        dfqdt = (fqref - fq)/tp[0]
        dzcdt = (zcref - zc)/tp[1]
        dxdt=vertcat(dpbhdt,dpwhdt,dqdt,dfqdt,dzcdt)
        yp=vertcat(pin,(H*Hc+self.par.H_lim[0]))
        
        self.c_eq_medicao=Function('c_eq_medicao',[x],[yp],['xn'],['yp'])
        self.eq_estado = Function('eq_estado',[x,u],[dxdt],['x','u'],['dxdt'])
        dxdt_0 = self.eq_estado(x, u)
        self.J=sum1(dxdt_0**2)
        
        # EDO
        # -------------------------------------------------
 
        
        opt={
            'ipopt':{
                'print_level':0,
                'acceptable_tol':1e-8,
                'acceptable_obj_change_tol':1e-6,
                'max_iter':50
                },
            'print_time':0,
            }

        MMQ = {'x':x, 'f':self.J, 'p':u}
        self.solver = nlpsol('solver', 'ipopt', MMQ, opt)
        args={
            'lbx': np.zeros((self.nx,1)),
        # m�ximo
            'ubx':np.full((self.nx, 1), np.inf)
            }


        
    def eq_medicao(self,x):
        pbh = x[0]
        q = x[2]
        fq = x[3]
        # Calculo do HEAD e delta de press�o
        q0 = q / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3;  # Potencia
        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * (q) ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * q)) ** (1 / 4)
        pin = pbh - rho * g * h1 - F1
        return vertcat([pin, H])
        
    def integrator_ode(self,xss,uss):
        xssn=(xss-self.par.x0)/self.par.xc
        #array([0.656882  , 0.58981816, 0.41643043])
        sol=self.solver(x0=xssn, p=uss)
        xn=np.array(sol['x'])
        x=np.array(xn*self.par.xc+self.par.x0)
        return x
    def c_integrator_ode(self,xss,uss):
        xssn=(xss-self.par.x0)/self.par.xc
        #array([0.656882  , 0.58981816, 0.41643043])
        sol=self.solver(x0=xssn, p=uss)
        xn=sol['x']
        x=xn*self.par.xc+self.par.x0
        return x
        


    
    def nmpc_solver(self,u0,utg,ypk,xmk,ymk2,pm,Du0,ymin,ymax):
        du0 = Du0

        ysp0 = ypk      
        
        Qu = np.diag(np.array([self.qu]))
        Q = np.diag(np.tile(self.q,(1,self.Hp))[0,:]); # dimension Hp*ny x Hp*ny 
        R = np.diag(np.tile(self.r,(1,self.Hc))[0,:]); # dimension Hc*nu x Hc*nu
        
        
        #print(xmk)
        x0=np.vstack([np.tile(ysp0,(self.Hp,1)), Du0])


        
        ysp=SX.sym('ysp',(self.Hp*self.ny,1))
        du=SX.sym('du',self.nu*self.Hc,1)
        
        
        for k in range(self.Hc):
            u0 = u0 + du[(k)*self.nu:k*self.nu+2,:]
        xv = vertcat(du,SX.zeros((self.nu*(self.Hp-self.Hc),1))) # du: decision variables Hc*nu
        ym = self.c_open_loop_sim(xmk,xv,u0,pm) 
                # Bias 
        e=ypk-ymk2; # mismatch at time step k
        ee=np.tile(e,(self.Hp,1))
        y=ym+ee; # augmented vector of prediction plus bias
        #print(ysp.shape,y.shape)
        
        M_ymin=np.tile(ymin,(self.Hp,1))
        M_ymax=np.tile(ymax,(self.Hp,1))
        M_dumax=np.tile(np.array([[self.Ts*0.5], [self.Ts*0.5]]),(self.Hc,1))


        g_bound=vertcat(-M_ymin+ysp, -ysp+M_ymax,du+M_dumax,-du+M_dumax)

        
        #Decision Variables
        du = Opti.variable(self.nu*self.Hc)
        ysp = Opti.variable(self.Hp*self.ny)
        #Objective function
        J = (((y-ysp).T@Q@(y-ysp)) + (du.T@R@du.printme(0)) + ((u0[1] - utg).T*Qu*(u0[1] - utg)))
        Opti.minimize(J)
        Opti.subject_to(-M_ymin+ysp>=0)
        Opti.subject_to(-ysp+M_ymax>=0)
        Opti.subject_to(du+M_dumax>=0)
        Opti.subject_to(-du+M_dumax>=0)
        Opti.solver('ipopt')
        sol = Opti.solve()
        qp = {'x':vertcat(ysp,du), 'f':J, 'g':g_bound}
        S = qpsol('S', 'qpoases', qp, {"printLevel": "none"})       
        r = S(x0=x0)
        
        #print(S.stats())
        x_opt = r['x']
        Du=x_opt[self.Hp*self.ny:self.Hp*self.ny+self.Hc*self.nu,:]
        ysp_opt=x_opt[:self.ny,:]
        return Du,ysp_opt
        
    def bcs_model(self,t,x,u):
        #estados
        pbh = x[0]
        pwh = x[1]
        q = x[2]
        fq = x[3]
        zc = x[4]
        # Entradas
        fqref = u[0]
        zcref = u[1]
        pm=self.pm
        pr=1.26e7
        # pm=u[2]
        # pr=u[3]
        
        # Calculo do HEAD e delta de press�o
        q0 = (q*self.par.qc+self.par.qmin) / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        #Pp = rho * g * H  # Delta de press�o

        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3;  # Potencia
        I = Inp * P / Pnp  # Corrente

        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        F2 = 0.158 * ((rho * L2 * ((q*self.par.qc+self.par.qmin)) ** 2) / (D2 * A2 ** 2)) * (mu / (rho * D2 * ((q*self.par.qc+self.par.qmin)))) ** (1 / 4)
        pin = pbh*self.par.pbc+self.par.pbmin - rho * g * h1 - F1
        # Vazao do reservatorio e vazao na choke
        qr = PI * (pr - (pbh*self.par.pbc+self.par.pbmin))
        qch = (zc/100)*Cc * math.sqrt(abs(pwh*self.par.pwc+self.par.pwmin - pm));

        # Termos não lineares
        # #menor q implica em menor F
        # funcH=Function('funcH',[self.x,self.u],[H])
        # funcF1=Function('funcF1',[self.x],[F1])
        # funcF2=Function('funcF2',[self.x],[F2])
        # #F1lim=(funcF1([0,0,self.par.qlim[0]]),funcF1([0,0,self.par.qlim[1]]))
        # #F2lim=(funcF2([0,0,self.par.qlim[0]]),funcF2([0,0,self.par.qlim[1]]))
        F1c=Lim_c(self.par.F1lim)
        F2c=Lim_c(self.par.F2lim)
        Hc=Lim_c(self.par.H_lim)
        qcc=Lim_c(self.par.qch_lim)
        #Normalizar termos não lineares
        ##########################
        qch=(qch-self.par.qch_lim[0])/qcc
        F1=(F1-self.par.F1lim[0])/F1c
        F2=(F2-self.par.F2lim[0])/F2c
        H=(H-self.par.H_lim[0])/Hc
        ###########################

        dpbhdt = (1/self.par.pbc)*b1/V1*(qr - (q*self.par.qc+self.par.qmin))
        dpwhdt = (1/self.par.pwc)*b2/V2*((q*self.par.qc+self.par.qmin) - (qcc*qch+self.par.qch_lim[0]))
        dqdt = (1/(self.par.qc*M))*(pbh*self.par.pbc+self.par.pbmin - (pwh*self.par.pwc+self.par.pwmin) - rho*g*hw - (F1c*F1+self.par.F1lim[0]) - (F2c*F2+self.par.F2lim[0]) + rho * g * (H*Hc+self.par.H_lim[0]))
        dfqdt = (fqref - fq)/tp[0]
        dzcdt = (zcref - zc)/tp[1]
        dudt_max = [dfq_max, dzc_max]
        dudt = np.zeros((1,2))
        if abs(dfqdt)>dudt_max[0]:
            dudt[0] = np.sign(dfqdt)*dudt_max[0]
        else:
            dudt[0] = dfqdt       
        if (abs(dzcdt)>dudt_max[1]):
            dudt[1] = np.sign(dzcdt)*dudt_max[1]
        else:
            dudt[1] = dzcdt   
        dxdt=np.vstack(dpbhdt,dpwhdt,dqdt,dudt[0],dudt[1])
        
        return dxdt
    def open_loop_sim(self,x0m,du,uk_1,pm):
        y=[]
        for k in range(self.Hp):
            uk_1 = uk_1 + du[(k)*self.nu:2+k*self.nu,:]
            xk=self.c_integrator_ode(x0m,uk_1)
            x0m = xk
            ymk = self.c_eq_medicao(x0m); # dimension ny  
            y.append(ymk) # dimension Hp*ny
        y=vertcat(y)
        return y
    def c_open_loop_sim(self,x0m,du,uk_1,pm):
        y=[]
        for k in range(self.Hp):
            uk_1 = uk_1 + du[(k)*self.nu:2+k*self.nu,:]
            xk=self.integrator_ode(x0m,uk_1)
            x0m = xk
            ymk = self.c_eq_medicao(x0m); # dimension ny  
            y.append(ymk) # dimension Hp*ny
        y=vertcat(y)
        return y
    
    def eq_medicao(self,x):
        pbh = x[0]
        q = x[2]
        fq = x[3]
        # Calculo do HEAD e delta de press�o
        q0 = q / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2  # Head
        # Calculo da Potencia e corrente da bomba
        P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
        P = Cp * P0 * (fq / f0) ** 3;  # Potencia
        # Calculo da press�o de intake
        F1 = 0.158 * ((rho * L1 * (q) ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * q)) ** (1 / 4)
        pin = pbh - rho * g * h1 - F1
        return np.vstack([pin, H])

