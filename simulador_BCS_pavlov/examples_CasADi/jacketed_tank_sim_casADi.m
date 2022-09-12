clear all
close all
clc
addpath('C:\Users\marcio\Downloads\casadi-windows-matlabR2016a-v3.5.1')
import casadi.*

% x = MX.sym('x'); z = MX.sym('z'); p = MX.sym('p');
% dae = struct('x',x,'z',z,'p',p,'ode',z+p,'alg',z*cos(z)-x);
% F = integrator('F', 'idas', dae);
% disp(F)
% r = F('x0',0,'z0',0,'p',0.1);
% disp(r.xf)

modelo.rho = 1e3       ;
modelo.Cp  = 4.18      ;
modelo.At  = pi*0.5^2  ;
modelo.k   = 7         ;
modelo.lambda = 2.257e4;
modelo.Ti = 40 ;

% Creating the symbolic variables
x = MX.sym('x',2); % Two state variables
u = MX.sym('u',2)  ; % parameters (input and/or model parameters)
% sx_function = mx_function.expand(x);
L = x(1);
T = x(2);
Fi = u(1);
Fv = u(2);

At = modelo.At;
k = modelo.k;
rho = modelo.rho;
Cp = modelo.Cp;
lambda = modelo.lambda;
Ti = modelo.Ti;

f1 = ( Fi - k*sqrt(L) ) / At;
f2 = ( rho*Fi*Cp*(Ti - T) +  Fv*lambda )/ ( rho*At*L*Cp ) ;
rhs = [f1; f2];  % ODE system defined in this variable rhs

f=Function('f',{x,u},{rhs},{'x','u'},{'rhs'}); % CasADi function (Function)

options = struct;
options.tf=0.1;  % integration step

% IDAS - DAE; cvodes - ODE (solvers)
ode = struct;     
ode.x   = x;      % states
ode.p   = u;      % inputs
% ode.ode = f(x,u); % right-hand side
ode.ode = rhs; % right-hand side
intg = integrator('intg','cvodes',ode, options);
res = intg('x0',x,'p',u);
x_next=res.xf;
F=Function('F',{x,u},{x_next},{'x','u'},{'x_next'}); % 1-step ahead function

% Estado estacionario na entrada
Fi = 10;
Fv = 10;

% Estado estacionario no estados
L = 2.0408;
T = 45.4  ;
x0=[L;T];

Ts=options.tf;
tsim=50;
nsim=tsim/Ts;
X(:,1)=x0;
for j=1:nsim
    if j <= 2/Ts
        Fi=10; Fv=10; u=[Fi;Fv];
    elseif j >2/Ts && j<=30/Ts
        Fi=15; Fv=10; u=[Fi;Fv];
    else
       Fi=15; Fv=10*1.1; u=[Fi;Fv];
    end
    tic
    X_next=F(X(:,j),u);
    X(:,j+1)=full(X_next);
    tcalc(j)=toc;
end
figure(1)
subplot(2,1,1)
plot(0:Ts:nsim*Ts,X(1,:))
xlabel('tempo /min')
ylabel('nivel /m')
grid on
subplot(2,1,2)
plot(0:Ts:nsim*Ts,X(2,:))
xlabel('tempo /min')
ylabel('Temperature /ºC')
grid on

figure(2)
bar(1:nsim,tcalc,'k')
title('Time at each time step')
axis([1 nsim 0 0.05])
