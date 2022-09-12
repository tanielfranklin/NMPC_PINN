
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math
#Car race control
#Variable to build model
N=100
op=cs.Opti()
x=cs.MX.sym('x',2)
u=cs.MX.sym('u',1)
opti = cs.Opti() # Optimization problem
X = opti.variable(2,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]
U = opti.variable(1,N+1)   # control trajectory (throttle)
T = opti.variable()  
rhs=cs.vertcat(x[0],u-x[1])
F_ode=cs.Function('F_ode',[x,u],[rhs],['x','u'],['rhs'])
dae = {'x':x, 'ode':F_ode(x,u), 'p':u} 
opts = {'tf':T/N} # decision variable in integration interval          
intg = cs.integrator('intg', 'cvodes', dae,opts)
res = intg(x0=x,p=u)
xk1=res['xf']
F=cs.Function('F',[x,u],[xk1],['x','u'],['xk1']) #Compute next step




N = 100 # number of control intervals
opti = cs.Opti() # Optimization problem
X = opti.variable(2,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]
U = opti.variable(1,N+1)   # control trajectory (throttle)
T = opti.variable()      # final time
rhs=cs.vertcat(X[0,:],U-X[1,:]) # right hand side of ode
F_ode=cs.Function('F_ode',[X,U],[rhs],['X','U'],['rhs'])
dxdt_0=F_ode(X,U)
dae = {'x':X, 'ode':rhs, 'p':U} # eq_ode with symbolic input
opts = {'tf':T/N} # decision variable in integration interval      

opt={
        'print_level':0,
        'acceptable_tol':1e-8,
        'acceptable_obj_change_tol':1e-6,
        'max_iter':50
    }
opti.minimize(J)
solver_int = opti.solver('ipopt')

sol = opti.solve()
Xopt=sol.value(X)

res = intg(x0=X,p=U) # solution with symbolic input
xk1=res['xf']
F=cs.Function('F',[U,X],[xk1],['pos','speed'],['xk1']) #Compute next step

f = lambda x,u: cs.vertcat(x[1],u-x[1]) # dx/dt = f(x,u)
for k in range(N):
    x_next = F(X[:,k+1],U[:,k+1])
    opti.subject_to(X[:,k+2]==x_next) # close the gaps
    
    

dt = T/N; # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],U[:,k])
   k2 = f(X[:,k]+dt/2*k1,U[:,k])
   k3 = f(X[:,k]+dt/2*k2,U[:,k])
   k4 = f(X[:,k]+dt*k3,U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4)
   opti.subject_to(X[:,k+1]==x_next) # close the gaps


limit = lambda pos: 1-cs.sin(2*3.14*pos)/2

opti.subject_to(speed<=limit(pos)); # track speed limit
opti.subject_to(U<=1);           # control is limited
opti.subject_to(0<=U);   
opti.subject_to(pos[1]==0);   # start at position 0 ...
opti.subject_to(speed[1]==0); # ... from stand-still
opti.subject_to(pos[N]==1); # finish line at position 1

opti.subject_to(T>=0); # Time must be positive
opti.minimize(T); # race in minimal time

#Provide initial guesses for the solver:

opti.set_initial(speed, 1);
opti.set_initial(T, 1);

#Solve the NLP using IPOPT

opti.solver('ipopt'); # set numerical backend
sol = opti.solve();   # actual solve

#Post processing of the optimal values.

plt.plot(sol.value(speed))
plt.plot(sol.value(pos))

plt.figure()
t = np.linspace(0,sol.value(T),N+1);
plt.plot(t,sol.value(speed),label="Speed");
plt.plot(t,sol.value(pos),label="Pos");
plt.plot(t,limit(sol.value(pos)),label="speed limit")
plt.stairs(sol.value(U),t, label="throttle");
plt.xlabel('Time [s]');
plt.legend()
print('OCP_sol','-dpng')
plt.show()