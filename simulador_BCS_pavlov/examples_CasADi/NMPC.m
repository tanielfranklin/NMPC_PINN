clear all
close all
clc

addpath('C:\Users\marcio\Downloads\casadi-windows-matlabR2016a-v3.5.1')
import casadi.*

% Model parameters
modelo.rho = 1e3       ;
modelo.Cp  = 4.18      ;
modelo.At  = pi*0.5^2  ;
modelo.k   = 7         ;
modelo.lambda = 2.257e4;
modelo.Ti = 40 ;

% Plant model parameters
modelo_p.rho = 1e3       ;
modelo_p.Cp  = 4.18      ;
modelo_p.At  = pi*0.5^2  ;
modelo_p.k   = 7         ;
modelo_p.lambda = 2.257e4;
modelo_p.Ti = 40 ;

%% CasADi function for ODEs 
% Creating the symbolic variables
x = MX.sym('x',2); % Two state variables
u = MX.sym('u',2)  ; % parameters (input and/or model parameters)

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
ode.ode = f(x,u); % right-hand side
intg = integrator('intg','cvodes',ode, options);
res = intg('x0',x,'p',u);
x_next=res.xf;
F=Function('F',{x,u},{x_next},{'x','u'},{'x_next'}); % 1-step ahead function

%% Controller parameters
nsim = 100 ; % number of steps
Hp = 10   ; % prediction horizon
Hc = 2   ; % control horizon
q = 10*[1 1] ; % weights on controlled variables 
Q = diag(q);
r = [0.1 0.1]  ; % weights on control actions
Rbar = diag(repmat(r,1,Hc)); % dimension Hc*nu x Hc*nu
Ts = 0.1; % sampling time
ny=size(q,2);
nu=size(r,2);

%--------------------------------------------------------------------------
% constraints on inputs
%--------------------------------------------------------------------------
umin  = [5 5]'   ;  % lower bounds of inputs
umax  = [20 30]'   ;  % upper bounds of inputs 
dumax = [1 2]' ;  % maximum variation of input moves
% Auxiliary matrices for using in the solver
Dumax = dumax;
for i = 1:Hc-1
  Dumax = [Dumax;dumax];
end
Dumin = -Dumax;
Mtil=[];
Itil=[];
auxM=zeros(nu,Hc*nu);
for in=1:Hc
    auxM=[eye(nu) auxM(:,1:(Hc-1)*nu)];
    Mtil=[Mtil;auxM];
    Itil=[Itil;eye(nu)];
end
% Ain = [Mtil;-Mtil];
% Bin = @(uk_1) [repmat(umax,Hc,1) - Itil*uk_1; Itil*uk_1 - repmat(umin,Hc,1)];
% Aeq = [];
% beq = [];

%% CasADi function for optimizer
% Creating the symbolic variables
du = MX.sym('du',Hc*nu)     ; % decision variables (control action sequence)
P = MX.sym('P',ny+nu+ny+ny)  ; % parameters (initial state, uk_1, set-point, state mismatch)
X = MX.sym('X',ny,Hp+1)     ; % predicted state trajectory over prediction horizon

g=[X(:,1)-P(1:ny)];
uk_1 = P(ny+1:nu+ny);
xsp=P(nu+ny+1:nu+ny+ny);
ee=P(nu+ny+ny+1:nu+ny+ny+ny);
xv = [du;zeros(nu*(Hp-Hc),1)]; % du: decision variables Hc*nu
obj_1 = 0; % objective function
for k=1:Hp
    uk_1 = uk_1 + xv((k-1)*nu+1:k*nu);
    X_next=F(X(:,k),uk_1);
    obj_1=obj_1+(X(:,i+1)+ee-xsp)'*Q*(X(:,i+1)+ee-xsp);
%     X(:,k+1)=X_next;
    g=[g;X(:,k+1)-X_next]; % model constraints (Hp.ny)
end

obj = obj_1+du'*Rbar*du; % objective function of NMPC
g = [g; Mtil*du+Itil*P(ny+1:nu+ny)] ; % including constraints on inputs (Hc.nu)

opt_variable=[X(:); du];
nlp_NMPC = struct('f',obj,'x',opt_variable,'g', g, 'p', P);
options=struct;
options.ipopt.max_iter=100;
options.ipopt.print_level=3; %0,3
options.print_time=3;
options.ipopt.acceptable_tol=1e-8;
options.ipopt.acceptable_obj_change_tol=1e-6; 
solver = nlpsol('solver', 'ipopt', nlp_NMPC, options);

% Arguments for solver
args=struct;
args.lbg(1:ny*(Hp+1),1) = 0; % bounds for equality constraints
args.ubg(1:ny*(Hp+1),1) = 0; % bounds for equality constraints
args.lbg(ny*(Hp+1)+1:nu:ny*(Hp+1)+nu*Hc,1) = umin(1); % lower bound for u(1)
args.lbg(ny*(Hp+1)+2:nu:ny*(Hp+1)+nu*Hc,1) = umin(2); % lower bound for u(2)
args.ubg(ny*(Hp+1)+1:nu:ny*(Hp+1)+nu*Hc,1) = umax(1); % upper bound for u(1)
args.ubg(ny*(Hp+1)+2:nu:ny*(Hp+1)+nu*Hc,1) = umax(2); % upper bound for u(2)

args.lbx(1:ny*(Hp+1),1) = 0  ; % lower bound for state x
args.ubx(1:ny*(Hp+1),1) = inf; % upper bound for state x
args.lbx(ny*(Hp+1)+1:nu:ny*(Hp+1)+nu*Hc,1) = -dumax(1); % lower bounds for \Delta u(1)
args.lbx(ny*(Hp+1)+2:nu:ny*(Hp+1)+nu*Hc,1) = -dumax(2); % lower bounds for \Delta u(2)
args.ubx(ny*(Hp+1)+1:nu:ny*(Hp+1)+nu*Hc,1) =  dumax(1); % upper bounds for \Delta u(1)
args.ubx(ny*(Hp+1)+2:nu:ny*(Hp+1)+nu*Hc,1) =  dumax(2); % upper bounds for \Delta u(2)

%--------------------------------------------------------------------------
% Initial condition (steady-state preferentially)
%--------------------------------------------------------------------------
uk = [10 10]'; % manipulated variable
u_cl = uk;
x0 = [2.0408 45.4]'; % state variables
x0m = [2.0408 45.4]' ;
%--------------------------------------------------------------------------
du0  = zeros(Hc*nu,1);  % Initial guess for decision variables (\Delta u_k)
X0 = repmat(x0,1,Hp+1); % Initial guess for decision variables (X_k)
xx(:,1)=x0;
xxm(:,1)=x0m;
e(:,1)=x0-x0m;
t0 = 0;
t(1)=t0;

for k = 1:nsim
    %     changes on set-points
    if k <= 300
        ysp = [3 50]';
    else
        ysp = [3 50]';
    end
    args.p = [x0m;uk;ysp;e(:,k)];
    args.x0 = [X0(:);du0]; 
    sol=solver('x0',args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    dU = full(sol.x(ny*(Hp+1)+1:end)); % sequence of control actions
    J_k(k) = full(sol.f) ; % cost function
    uk = uk + dU(1:nu); % optimal input at time step k
    u_cl=[u_cl uk]; % storing optimal inputs
 
    % Plant (Matlab or gProms or Fortran or C, and so on)
    if k <= 80
       xx_next=F(x0,uk);
    else
       xx_next=F(x0,uk*1.1);
    end

    xx(:,k+1) = full(evalf(xx_next))+0*mvnrnd(zeros(1,ny),diag([.01 .01]))';  
    x0 = xx(:,k+1);
    
    % Model (ANN, ODE, DAE)
    xxm_next=F(x0m,uk);
    xxm(:,k+1) = full(evalf(xxm_next));
    x0m = xxm(:,k+1);
    
    % update of mismatch
    e(:,k+1)=x0-x0m;
    
    % initial estimate for optimizer (warm start)
     du0 = [dU(nu+1:end);zeros(nu,1)];
     X0=reshape(full(sol.x(1:ny*(Hp+1)))',ny,Hp+1);
     sp(:,k) = ysp;
end

figure(1)
for in = 1:ny
    subplot(ny,1,in)
    plot(0:Ts:nsim*Ts,xx(in,:),'k',Ts:Ts:nsim*Ts,sp(in,:),'r--')
    ylabel(['y_' num2str(in)])
    grid on
end
xlabel('Time /(min)')
% legend('Setpoint','Medicao','Modelo','Location','Best')

figure(2)
for iu = 1:nu
    subplot(nu,1,iu)
    stairs(0:Ts:nsim*Ts,u_cl(iu,:),'k')
    ylabel(['u_' num2str(iu)])
    grid on
end
xlabel('Time /(min)')

figure(3)
plot(Ts:Ts:nsim*Ts,J_k,'k')
xlabel('Time /(min)')
ylabel('Cost function')

figure(4)
plot(0:Ts:nsim*Ts,e,'k')
xlabel('Time /(min)')
ylabel('Erro')

% figure(5)
% bar(1:nsim,tcalc,'k')
% title('Time at each time step')
% axis([1 nsim 0 inf])
% ylabel('Erro')