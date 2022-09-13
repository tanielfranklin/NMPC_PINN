import numpy as np
import casadi as cs
from casadi import vertcat, MX, Opti, Function, sin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from BCS_casadi import BCS_model
from plot_result import PlotResult
from bcs_envelope import BcsEnvelope
from nmpc_class import NMPC



# assim o tempo computacional do solver reduz consideravelmente!!
# codegen -lang:c++ solverNMPC -args {uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,pm,DuYsp0,Ain,Bin,Aeq,beq,Dumin,Dumax} -report
# steady-state conditions
xss = np.vstack([8311024.82175957, 2990109.06207437,
                0.00995042241351780, 50., 50.])

nx = 5
nu = 2

uss = np.vstack([50., 50.])
yss = np.vstack([6000142.88550200, 592.126490003812])
# Controller parameters
Hp = 10  # prediction horizon 
Hc = 2  # control horizon
Ts = 2  # sampling time
# --------------------------------------------------------------------------
# constraints on inputs
# --------------------------------------------------------------------------
umin = np.vstack([35, 0])    # lower bounds of inputs
umax = np.vstack([65, 100])    # upper bounds of inputs
# maximum variation of input moves: [0.5 Hz/s; 0.5 #/s]
dumax = Ts*np.vstack([0.5, 0.5])
# ----------------------------------------------
# ------Normalização através dos pesos----------
q = np.hstack([100, 1]) / (yss**2)  # weights on controlled variables
q = np.hstack([1e6, 1e8]) / (yss**2)  # weights on controlled variables

r = np.array([100, 1]) / (uss.T**2)  # weights on control actions
# r = np.array([10, 10]) /(uss.T**2); # weights on control actions

qu = 1000 / (uss[1]**2)
ny = 2
print("Instancia BCS")
bcs_init = [nu, nx, ny, Ts, umin, umax, dumax]
nmpc = NMPC(Hp, Hc, q, r, qu, bcs_init)
bcs = nmpc.bcs
# ukk=bcs.par.xc,bcs.par.x0
# (array([1.25000000e+07, 4.90000000e+06, 1.38888889e-02]),
# array([1.00000000e+05, 1.00000000e+05, 4.16666667e-03]))
# --------------------------------------------------------------------------
# Initial condition (preferred steady-state)
# --------------------------------------------------------------------------
uk_1 = np.vstack([50., 50.])  # manipulated variable
x0 = np.vstack([8311024.82175957, 2990109.06207437,
               0.00995042241351780, 50., 50.])  # state variables
xss = bcs.integrator_ode(x0, uk_1)
xssn=bcs.norm_x(xss)
x0n = np.vstack([0.656882, 0.58981816, 0.41643043, 50.0, 50.0])
xss = bcs.integrator_ode(xss, uk_1)
xssn = (xss-bcs.par.x0)/bcs.par.xc

print(xssn.T)
# print(x0n.T)
utg = 50.
yss2 = bcs.c_eq_medicao(xssn)
u0 = uss
xmk = xss
Du = np.zeros((nmpc.Hc*bcs.nu, 1))
ysp = yss



ypk = yss
xmk = x0
ymk = yss  # condição inicial para o EKF
xmk[2] = 0.0106  # inicia a vazao de um x0 diferente para testar converg.
xmk2 = x0
ymk2 = yss  # estados da simulação do modelo nominal
utg = 70   # target na choke
pm = 2e6   # pressão da manifold
# ymax[0,0] = yss[0]; # Pressao de intake
# Regiao de operação
hlim = bcs.envelope.Hlim(x0[2]*3600)
ymin = np.vstack([yss[0], min(hlim)])  # Pressao de intake e  Downtrhust
ymax = np.vstack([xss[0], max(hlim)])  # Pressao de intake e  Uptrhust
# ymin(2,1) = min(hlim); # Downtrhust
# ymax(2,1) = max(hlim); # Uptrhust
xpk = xss

Du = np.ones((nmpc.Hc*bcs.nu, 1))
ysp = yss
# --- simulação -------------------------------- ------------------------------
tsim = 50     # seconds
nsim = int(tsim/Ts)   # number of steps
uk = np.zeros((bcs.nu, int(nsim)))
Vruido = ((0.01/3)*np.diag(yss[:, 0]))**2

Du = np.zeros((nmpc.Hc*bcs.nu, 1))
# # Parameters: initial states, du,utg, u0,ysp
P = np.vstack([x0, uk_1, Du, utg])
#x0 u0 du utg
Du, ysp=nmpc.nmpc_solver(P, ymin, ymax)
print(Du)
print(ysp)

Yk = yss
Xk = xss.reshape((xss.shape[0], 1))
Ysp = yss
YLim = np.vstack([ymax[1], ymin[1]])
for k in range(nsim):
    print("Iteração:", k)
    tsim = k*Ts
    # changes on set-points Pintake
    # if k==50:

    #     ymin[0,0] = 5e5
    # elif k==100:
    #     ymin[0,0] = 5e6

    ymax[0, 0] = ymin[0, 0]
    # elif k==200:
    #    ymin[0,0] = 4.2e6
    # else:
    #     bcs.pm = 8e5
    # if k==20:
    #     utg=80
    #     #uk_1[0,0]=55
    # if k==60:
    #     utg=90

    # #ymin(1,1) = yss(1);    # Pressao de intake
    # ymax[0,0] = ymin[0,0]; # Pressao de intake

    # #ymin[0,0] = 4e6;
    # ## Limite Up e Downthrust
    # hlim = bcs.envelope.Hlim(xpk[2]*3600)
    ymin[1, 0] = min(hlim)
    ymax[1, 0] = max(hlim)

    # tic
    
    Du, ysp = nmpc.nmpc_solver(P, ymin, ymax)
    uk[:, k:k+1] = uk_1 + Du[:nmpc.Hc, :]
    uk_1 += Du[:nmpc.Hc, :]  # optimal input at time step k
    #update input vector with the states and Du
    P = np.vstack([x0, uk_1, Du, utg])
    #P = np.vstack([x0, Du, utg, uk_1, Du, yss])

    # J_k[k] = fval  # cost function

    # iter[0,k] = report.iterations
    # evalObj[1,k] = report.funcCount
    # flags[1,k] = flag
    # DuYsp0 = DuYsp # Estimativa inicial do otmizador
    # print(uk)
    # Plant

    xpk = bcs.integrator_ode(x0, uk_1)
    #print(xpk)
    #  Nominal Model
    x0 = xpk
    ypk = bcs.eq_medicao(x0)
    ruido = 5
    # +np.random.multivariate_normal((0,0), Vruido).reshape((bcs.ny,1)) # ruido -> 3x a 5x
    ypk = ypk
    Yk = np.concatenate((Yk, ypk), axis=1)
    Ysp = np.concatenate((Ysp, ysp), axis=1)
    YLim = np.concatenate((YLim, np.vstack([ymax[1], ymin[1]])), axis=1)
    Xk = np.concatenate((Xk, x0.reshape((x0.shape[0], 1))), axis=1)


grafico = PlotResult()
grafico.plot_resultado(Xk, uk)
grafico.plot_y(Ysp, Yk, YLim)
bcs.envelope.size_env = (4, 4)
bcs.envelope.grafico_envelope(Xk, Yk)
plt.show()

exit()

nx = nmpc.bcs.nx
ny = nmpc.bcs.ny
nu = nmpc.bcs.nu
Qu = np.diag(np.array([nmpc.qu]))  # Weights control effort 
Q = np.diag(np.array([nmpc.q[0, :]]))  # Weights setpoint error
R = np.diag(np.tile(nmpc.r, (1, nmpc.Hc))[0, :]) # Weights economic target error
opti = nmpc.opti
# define decision variables
du = opti.variable(4)
ysp = opti.variable(2)
x0 = P[:nx]  # Get X0 from P
print('x0'); print(x0.T)

x0n = bcs.norm_x(x0)  # Normalize states x0
print('x0n'); print(x0n.T)

len_du = nu*nmpc.Hc
utg = P[nx+len_du:nx+len_du+1,0]  # Get economic target from P
print('utg',utg)

# Get initial control actions from P
ysp0 = P[-ny:, :]
print('ysp0'); print(ysp0.T)

ysp02 = bcs.c_eq_medicao(x0n)
du0 = P[nx+len_du+1+nu:nx+len_du+1+nu+len_du]
print('du0'); print(du0.T)

# Initialize decision variables
opti.set_initial(ysp, ysp0)
opti.set_initial(du, du0)
# Recovering predictions of states and outputs matrices
# X, Y, u = nmpc.FF(du, P)
# print('Matriz de predição inicial')
# X, Y, u = nmpc.FF(du0, P)
# print(X)
X, Y, u = nmpc.FF(du, P)


# #Define dynamic constraints  dependent of predictions steps
for k in range(nmpc.Hp):
    # opti.subject_to(X[:, k+1] == X[:, k])
    opti.subject_to(Y[:, k+1] >= ymin)
    opti.subject_to(Y[:, k+1] <= ymax)
# #Define contraints related to maximum and minimum du rate

opti.subject_to(np.tile(nmpc.bcs.dumax, (nu, 1))
                >= du)  # Maximum control rate
opti.subject_to(-np.tile(nmpc.bcs.dumax, (nu, 1))
                <= du)  # Minimun control rate



obj_1, obj_2 = 0, 0
for k in range(nmpc.Hp):
    # Track ysp_opt
    obj_1 = obj_1+(Y[:, k]-ysp).T@Q@(Y[:, k]-ysp)
    # Track Maximum zc
    obj_2 = obj_2+((u[1] - utg).T*Qu*(u[1] - utg))

#obj=obj_1.printme(0)+ obj_2.printme(1)+(du.T@R@du).printme(2)

obj = obj_1+ obj_2+(du.T@R@du)
#nmpc.Fobj=cs.Function('Fobj',[du,ysp],[nmpc.obj],['du','ysp'],['obj'])


opti.minimize(obj)
p_opts = {"expand": True}
s_opts = {"max_iter": 80, "print_level": 3}
#opti.solver("ipopt", p_opts, s_opts)
opti.solver("ipopt")
sol = opti.solve()


Du0 = np.zeros((nmpc.Hc*bcs.nu, 1))
# # Parameters: initial states, du,utg, u0,ysp
P = np.vstack([x0, Du0, utg, uk_1, Du, yss])
Du = nmpc.opti.variable(4)

print((nmpc.FF(Du, P)))
Du = sol.value(du)
ysp_opt = sol.value(ysp)[:ny]
print(Du,ysp_opt)
#nmpc.nmpc_solver(P, ymin, ymax)  # how
#print((nmpc.Fobj(Du,ysp)))
exit()






