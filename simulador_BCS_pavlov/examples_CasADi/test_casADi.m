clear all
clc
addpath('C:\Users\marcio\Downloads\casadi-windows-matlabR2016a-v3.5.1')
import casadi.*

% Creating the symbolic variables
x = MX.sym('x',2); % Two state variables
u = MX.sym('u')  ; % parameters (input and/or model parameters)
rhs =[(1-x(2)^2)*x(1)-x(2)
       x(1)               ] ; % ODE system defined in this variable rhs
f=Function('f',{x,u},{rhs},{'x','u'},{'rhs'}); % CasADi function (Function)

options = struct;
options.tf=0.1;  % integration step
% options.grid=linspace(0,4,100);

% IDAS - DAE; cvodes - ODE (solvers)
ode = struct;     
ode.x   = x;      % states
ode.p   = u;      % inputs
ode.ode = f(x,u); % right-hand side
intg = integrator('intg','cvodes',ode, options);
% x=[0;1]; u=0;
res = intg('x0',x,'p',u);
xx = full(res.xf);
x_next=res.xf;
F=Function('F',{x,u},{x_next},{'x','u'},{'x_next'}); % 1-step ahead function
tsim = 4; % final time [=]s
nsim=tsim/options.tf;
x0=[0;1]; u0=0;
X(:,1)=x0;
for i=1:nsim
    X_next=F(X(:,i),u0);
    X(:,i+1)=full(X_next);
end


% % Start from x=[0;1]
% x0=[0;1];
% X0=x0;
% tsim = 4; % final time [=]s
% nsim=tsim/Ti;
% for i=1:nsim
%     tic
%     res = F('x0',x0);
%     toc
%     X(:,i) = full(res.xf);
%     x0 = res.xf;
% end
% 
nc=size(X,1);
figure(1)
for j=1:nc
    subplot(nc,1,j)
%     plot(0:Ti:Ti*nsim,[X0(j,:) X(j,:)],'k-')
    plot(0:options.tf:options.tf*nsim,X(j,:),'k-')
    xlabel('tempo /s')
    in = num2str(j);
    yrot = ['x_' in];
    ylabel(yrot)
end
% 
% % x = SX.sym('x'); z = SX.sym('z'); p = SX.sym('p');
% % dae = struct('x',x,'z',z,'p',p,'ode',z+p,'alg',z*cos(z)-x);
% % F = integrator('F', 'idas', dae);
% disp(F)
% r = F('x0',0,'z0',0,'p',0.1);
% disp(r.xf)