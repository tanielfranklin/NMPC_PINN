clear all
clc
addpath('C:\Users\marcio\Downloads\casadi-windows-matlabR2016a-v3.5.1')
import casadi.*

x = MX.sym('x',4);
u = MX.sym('u',2);
vx=x(1); vy=x(2); px=x(3); py=x(4);
theta=u(1); tf=u(2);
rhs = [0;-9.81;x(1);x(2)].*tf;
f=Function('f',{x,u},{rhs},{'x','u'},{'rhs'});
ode=struct;
ode.x=x; % state variables
ode.p=u; % model parameters
ode.ode=f(x,u);

% options=struct;
% options.tf=3.5;  % integration step
% options.grid=linspace(0,1,100);

F = integrator('F','cvodes',ode);
v0=20;
% theta_0=45;
x0=[v0*cos(theta);v0*sin(theta);0;1];
p=[theta;tf];
sol = F('x0',x0,'p',p);
X = full(sol.xf)';
% plot(X(:,3),X(:,4))

%% Optimization problem
% Maximize distance px
x_opt = [x;u];
prob = struct('x',x_opt,'f',-x(3),'g',x-sol.xf);
solver=nlpsol('solver','ipopt',prob);

sol_opt=solver('x0',[1;1;5;0;pi/6;5],...
           'lbg',[0;0;0;0],...
           'ubg',[0;0;0;0],...
           'lbx',[0;-inf;0;0;0;0],...
           'ubx',[inf; inf; inf; inf; pi/2; inf]);