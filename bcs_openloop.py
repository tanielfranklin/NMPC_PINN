import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.BCS_casadi import BCS_model
from data.plot_result import PlotResult
from data.bcs_envelope import BcsEnvelope
import seaborn as sns
sns.set_theme()

# steady-state conditions
xss = np.vstack([8311024.82175957, 2990109.06207437,
                0.00995042241351780, 50., 50.])
nx = 5
nu = 2
ny = 2

uss = np.vstack([50., 50.])
yss = np.vstack([6000142.88550200, 592.126490003812])
# output scale factor

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
qu = 1000 / (uss[1]**2)

print("Instancia BCS")
bcs_init = [nu, nx, ny, Ts, umin, umax, dumax]

bcs = BCS_model(nu,nx,ny,Ts,umin,umax,dumax)

# --------------------------------------------------------------------------
# Initial condition (preferred steady-state)
# --------------------------------------------------------------------------
uk_1 = np.vstack([50., 50.])  # manipulated variable
print("Initial states")
print("Pbh, Pwh, q, df, dzc")
x0 = np.vstack([8311024.82175957, 2990109.06207437,
               0.00995042241351780, 50., 50.])  # state variables
print("Initial control")
print("f, zc")

try:
    print(bcs.desnorm_x(bcs.F(x0=bcs.norm_x(x0), p=uss)['xf']))
except Exception as e:
    print("An exception was raised here:")
    print(e)



pm = 2e6   # pressão da manifold

# Regiao de operação
hlim = bcs.envelope.Hlim(x0[2]*3600)
ymin = np.vstack([yss[0], min(hlim)])  # Pressao de intake e  Downtrhust
ymax = np.vstack([xss[0], max(hlim)])  # Pressao de intake e  Uptrhust
# ymin(2,1) = min(hlim); # Downtrhust
# ymax(2,1) = max(hlim); # Uptrhust
xpk = xss
ysp = yss
Vruido = ((0.01/3)*np.diag(yss[:, 0]))**2

# # Parameters: initial states, du,utg, u0,ysp


Yk = yss
Xk = xss.reshape((xss.shape[0], 1))
Ysp = yss
HLim = np.vstack([ymax[1], ymin[1]])
PINLim = np.vstack([ymax[0], ymin[0]])
# Simulation Loop -------------------------------- ------------------------------
tsim = 6.    # minutes
nsim = int(60*tsim/Ts)   # number of steps
uk = np.zeros((bcs.nu, int(nsim)))
rows = []


for k in range(nsim):
    print("Iteração:", k)
    tsim = k*Ts/60

    # changes on set-points Pintake
    if tsim == 0.4:
        uk_1=np.vstack([50., 70.])
    elif tsim == 1.4:
        uk_1 = np.vstack([65., 70.])


    # elif k==200:
    #    ymin[0,0] = 4.2e6
    # else:
    #     bcs.pm = 8e5
    # if k==20:
    #     utg=80
    #     #uk_1[0,0]=55
    # if k==int(50/Ts):
    #     utg=90

    # #ymin(1,1) = yss(1);    # Pressao de intake
    # ymax[0,0] = ymin[0,0]; # Pressao de intake

    # #ymin[0,0] = 4e6;
    # ## Limite Up e Downthrust
    hlim = bcs.envelope.Hlim(xpk[2]*3600)
    ymin[1, 0] = min(hlim)
    ymax[1, 0] = max(hlim)

    uk[:, k:k+1] = uk_1 

    # Plant
    xpk = bcs.integrator_ode(x0, uk_1)

    #  Nominal Model
    x0 = xpk
    ypk = bcs.eq_medicao(x0)
    ruido = 5
    # +np.random.multivariate_normal((0,0), Vruido).reshape((bcs.ny,1)) # ruido -> 3x a 5x
    ypk = ypk
    Yk = np.concatenate((Yk, ypk), axis=1)
    Ysp = np.concatenate((Ysp, ysp), axis=1)
    HLim = np.concatenate((HLim, np.vstack([ymax[1], ymin[1]])), axis=1)
    PINLim = np.concatenate((PINLim, np.vstack([ymax[0], ymin[0]])), axis=1)
    Xk = np.concatenate((Xk, x0.reshape((x0.shape[0], 1))), axis=1)

grafico = PlotResult(bcs.Ts)
grafico.plot_resultado(Xk, uk)
grafico.plot_y(Ysp, Yk, HLim, PINLim)
bcs.envelope.size_env = (4, 4)
bcs.envelope.grafico_envelope(Xk, Yk)
plt.show()
